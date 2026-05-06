#pragma once
// ============================================================
//  include/distributed.h - Tiny native RPC surface for workers
// ============================================================

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <functional>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct DistributedConfig
{
      std::string mode{"none"};
      std::string role{"coordinator"};
      std::string worker_host{"127.0.0.1"};
      int worker_port{9091};
      std::string worker_token;
      std::string workers;
      std::string workers_file;
      int sync_interval{1};
      int gradient_bits{32};
      std::string shards{"auto"};
      bool coordinator_compute{true};
      int rpc_timeout_sec{900};
      int worker_reprobe_interval{5};
};

namespace quadtrix_dist_detail
{
static const uint32_t kMagic = 0x51445250u; // QDRP
static const uint16_t kVersion = 1;

enum MsgType
{
      Hello = 1,
      Status = 2,
      Error = 3,
      Stop = 4,
      TrainStep = 5,
      TrainResult = 6,
      DatasetChunk = 7,
      DatasetAck = 8,
      DatasetCheck = 9,
      DatasetStatus = 10,
      TokenizeJob = 11,
      TokenizeResult = 12,
      TokenCacheChunk = 13,
      TokenCacheAck = 14,
      TokenCacheCheck = 15,
      TokenCacheStatus = 16
};

struct Frame
{
      uint16_t type{0};
      uint32_t request_id{0};
      std::string payload;
};

inline bool write_all(int fd, const void *data, size_t n)
{
      const char *p = (const char *)data;
      while (n > 0)
      {
            ssize_t w = send(fd, p, n, 0);
            if (w <= 0) return false;
            p += w;
            n -= (size_t)w;
      }
      return true;
}

inline bool read_all(int fd, void *data, size_t n)
{
      char *p = (char *)data;
      while (n > 0)
      {
            ssize_t r = recv(fd, p, n, 0);
            if (r <= 0) return false;
            p += r;
            n -= (size_t)r;
      }
      return true;
}

inline bool send_frame(int fd, const Frame &frame)
{
      uint32_t magic = htonl(kMagic);
      uint16_t version = htons(kVersion);
      uint16_t type = htons(frame.type);
      uint32_t req = htonl(frame.request_id);
      uint32_t len = htonl((uint32_t)frame.payload.size());
      return write_all(fd, &magic, sizeof(magic)) &&
             write_all(fd, &version, sizeof(version)) &&
             write_all(fd, &type, sizeof(type)) &&
             write_all(fd, &req, sizeof(req)) &&
             write_all(fd, &len, sizeof(len)) &&
             (frame.payload.empty() || write_all(fd, frame.payload.data(), frame.payload.size()));
}

inline bool recv_frame(int fd, Frame &frame)
{
      const uint32_t max_payload = 512u * 1024u * 1024u;
      uint32_t magic = 0, req = 0, len = 0;
      uint16_t version = 0, type = 0;
      if (!read_all(fd, &magic, sizeof(magic)) ||
          !read_all(fd, &version, sizeof(version)) ||
          !read_all(fd, &type, sizeof(type)) ||
          !read_all(fd, &req, sizeof(req)) ||
          !read_all(fd, &len, sizeof(len)))
            return false;
      magic = ntohl(magic);
      version = ntohs(version);
      type = ntohs(type);
      req = ntohl(req);
      len = ntohl(len);
      if (magic != kMagic || version != kVersion || len > max_payload)
            return false;
      frame.type = type;
      frame.request_id = req;
      frame.payload.assign((size_t)len, '\0');
      return len == 0 || read_all(fd, &frame.payload[0], len);
}

inline std::string json_escape(const std::string &s)
{
      std::string out;
      for (char c : s)
      {
            if (c == '\\') out += "\\\\";
            else if (c == '"') out += "\\\"";
            else if (c == '\n') out += "\\n";
            else if (c == '\r') out += "\\r";
            else out += c;
      }
      return out;
}

inline std::vector<std::string> split_workers(const std::string &workers)
{
      std::vector<std::string> out;
      size_t pos = 0;
      while (pos <= workers.size())
      {
            size_t comma = workers.find(',', pos);
            if (comma == std::string::npos) comma = workers.size();
            std::string part = workers.substr(pos, comma - pos);
            if (!part.empty()) out.push_back(part);
            if (comma == workers.size()) break;
            pos = comma + 1;
      }
      return out;
}

inline int connect_to(const std::string &host, int port, int timeout_ms = 3000)
{
      int fd = socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0) return -1;
      sockaddr_in addr;
      std::memset(&addr, 0, sizeof(addr));
      addr.sin_family = AF_INET;
      addr.sin_port = htons((uint16_t)port);
      if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
      {
            close(fd);
            return -1;
      }

      int flags = fcntl(fd, F_GETFL, 0);
      bool nonblocking = timeout_ms > 0 && flags >= 0;
      if (nonblocking)
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);

      int rc = connect(fd, (sockaddr *)&addr, sizeof(addr));
      if (rc < 0 && errno == EINPROGRESS && nonblocking)
      {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(fd, &wfds);
            timeval tv;
            tv.tv_sec = timeout_ms / 1000;
            tv.tv_usec = (timeout_ms % 1000) * 1000;
            rc = select(fd + 1, nullptr, &wfds, nullptr, &tv);
            if (rc > 0)
            {
                  int so_error = 0;
                  socklen_t len = sizeof(so_error);
                  if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) != 0 || so_error != 0)
                        rc = -1;
                  else
                        rc = 0;
            }
            else
            {
                  rc = -1;
            }
      }

      if (nonblocking)
            fcntl(fd, F_SETFL, flags);

      if (rc < 0)
      {
            close(fd);
            return -1;
      }
      return fd;
}

inline bool parse_host_port(const std::string &s, std::string &host, int &port)
{
      size_t colon = s.find_last_of(':');
      if (colon == std::string::npos) return false;
      host = s.substr(0, colon);
      port = std::atoi(s.substr(colon + 1).c_str());
      return !host.empty() && port > 0;
}

inline void set_socket_io_timeout(int fd, int timeout_sec)
{
      if (timeout_sec <= 0) return;
      timeval tv;
      tv.tv_sec = timeout_sec;
      tv.tv_usec = 0;
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

inline std::string sanitize_header_value(const std::string &value)
{
      std::string out = value;
      for (char &c : out)
      {
            if (c == '\n' || c == '\r')
                  c = ' ';
      }
      return out;
}

inline std::string sanitize_path_component(const std::string &value)
{
      std::string out;
      for (char c : value)
      {
            unsigned char u = (unsigned char)c;
            if (std::isalnum(u) || c == '.' || c == '_' || c == '-')
                  out += c;
            else
                  out += '_';
      }
      return out.empty() ? "dataset" : out;
}

inline std::string base_name(const std::string &path)
{
      size_t pos = path.find_last_of("/\\");
      if (pos == std::string::npos) return path;
      return path.substr(pos + 1);
}

inline bool ensure_dir(const std::string &dir)
{
      if (dir.empty() || dir == ".") return true;
      if (mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST) return true;
      return false;
}

inline bool file_exists(const std::string &path)
{
      struct stat st;
      return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

inline std::string worker_dataset_cache_dir()
{
      const char *env = std::getenv("QUADTRIX_WORKER_DATASET_DIR");
      if (env && env[0]) return env;
      return "worker_datasets";
}

inline std::string worker_dataset_cache_path(const std::string &cache_key,
                                             const std::string &original_path)
{
      return worker_dataset_cache_dir() + "/" +
             sanitize_path_component(cache_key) + "_" +
             sanitize_path_component(base_name(original_path));
}

inline std::string token_cache_path_for_worker(const std::string &cache_dir,
                                               const std::string &cache_key)
{
      std::string dir = cache_dir.empty() ? "token_cache" : cache_dir;
      return dir + "/qwen3_" + sanitize_path_component(cache_key) + ".qtok";
}

inline std::string make_header_payload(const std::vector<std::pair<std::string, std::string>> &fields,
                                       const std::string &body)
{
      std::ostringstream out;
      for (const auto &kv : fields)
            out << kv.first << "=" << sanitize_header_value(kv.second) << "\n";
      out << "\n";
      std::string header = out.str();
      header += body;
      return header;
}

inline std::map<std::string, std::string> parse_header_payload(const std::string &payload,
                                                               size_t &body_offset)
{
      std::map<std::string, std::string> out;
      size_t header_end = payload.find("\n\n");
      if (header_end == std::string::npos)
      {
            body_offset = payload.size();
            return out;
      }

      size_t pos = 0;
      while (pos < header_end)
      {
            size_t nl = payload.find('\n', pos);
            if (nl == std::string::npos || nl > header_end) nl = header_end;
            size_t eq = payload.find('=', pos);
            if (eq != std::string::npos && eq < nl)
                  out[payload.substr(pos, eq - pos)] = payload.substr(eq + 1, nl - eq - 1);
            pos = nl + 1;
      }
      body_offset = header_end + 2;
      return out;
}

inline std::string handle_dataset_chunk_payload(const std::string &payload)
{
      size_t body_offset = 0;
      std::map<std::string, std::string> header = parse_header_payload(payload, body_offset);
      if (header.empty() || body_offset > payload.size())
            throw std::runtime_error("[DIST] Dataset payload is missing headers."
                                     "\n[ES] Al payload de dataset le faltan cabeceras.");

      std::string original_path = header["data_path"];
      std::string cache_key = header["dataset_cache_key"];
      int chunk_index = std::atoi(header["chunk_index"].c_str());
      int chunk_count = std::max(1, std::atoi(header["chunk_count"].c_str()));
      long long total_bytes = std::atoll(header["total_bytes"].c_str());
      if (original_path.empty() || cache_key.empty() || chunk_index < 0 ||
          chunk_index >= chunk_count || total_bytes < 0)
            throw std::runtime_error("[DIST] Dataset chunk metadata is invalid."
                                     "\n[ES] La metadata del chunk de dataset no es valida.");

      if (!ensure_dir(worker_dataset_cache_dir()))
            throw std::runtime_error("[DIST] Cannot create worker dataset cache directory."
                                     "\n[ES] No se puede crear la carpeta cache de datasets del worker.");

      std::string final_path = worker_dataset_cache_path(cache_key, original_path);
      std::string part_path = final_path + ".part";
      const char *mode = (chunk_index == 0) ? "wb" : "ab";
      FILE *f = std::fopen(part_path.c_str(), mode);
      if (!f)
            throw std::runtime_error("[DIST] Cannot write shared dataset cache."
                                     "\n[ES] No se puede escribir el cache de dataset compartido.");

      const std::string chunk = payload.substr(body_offset);
      if (!chunk.empty() && std::fwrite(chunk.data(), 1, chunk.size(), f) != chunk.size())
      {
            std::fclose(f);
            throw std::runtime_error("[DIST] Dataset cache write failed."
                                     "\n[ES] Fallo la escritura del cache de dataset.");
      }
      std::fclose(f);

      if (chunk_index + 1 == chunk_count)
      {
            if (std::rename(part_path.c_str(), final_path.c_str()) != 0)
                  throw std::runtime_error("[DIST] Cannot finalize shared dataset cache."
                                           "\n[ES] No se puede finalizar el cache de dataset compartido.");
      }

      std::ostringstream out;
      out << "{\"ok\":true,\"chunk\":" << chunk_index
          << ",\"chunks\":" << chunk_count
          << ",\"bytes\":" << chunk.size()
          << ",\"totalBytes\":" << total_bytes
          << ",\"path\":\"" << json_escape(final_path) << "\"}";
      return out.str();
}

inline std::string handle_dataset_check_payload(const std::string &payload)
{
      size_t body_offset = 0;
      std::map<std::string, std::string> header = parse_header_payload(payload, body_offset);
      if (header.empty())
            throw std::runtime_error("[DIST] Dataset check payload is missing headers."
                                     "\n[ES] Al payload de revision de dataset le faltan cabeceras.");

      std::string original_path = header["data_path"];
      std::string cache_key = header["dataset_cache_key"];
      long long expected_bytes = std::atoll(header["total_bytes"].c_str());
      if (original_path.empty() || cache_key.empty())
            throw std::runtime_error("[DIST] Dataset check metadata is invalid."
                                     "\n[ES] La metadata de revision de dataset no es valida.");

      std::string final_path = worker_dataset_cache_path(cache_key, original_path);
      struct stat st;
      bool readable = stat(final_path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
      long long bytes = readable ? (long long)st.st_size : -1;
      bool matches = readable && (expected_bytes < 0 || bytes == expected_bytes);

      std::ostringstream out;
      out << "{\"ok\":true,\"exists\":" << (matches ? "true" : "false")
          << ",\"bytes\":" << bytes
          << ",\"expectedBytes\":" << expected_bytes
          << ",\"path\":\"" << json_escape(final_path) << "\"}";
      return out.str();
}

inline std::string handle_token_cache_chunk_payload(const std::string &payload)
{
      size_t body_offset = 0;
      std::map<std::string, std::string> header = parse_header_payload(payload, body_offset);
      if (header.empty() || body_offset > payload.size())
            throw std::runtime_error("[DIST] Token-cache payload is missing headers."
                                     "\n[ES] Al payload de cache de tokens le faltan cabeceras.");

      std::string cache_key = header["token_cache_key"];
      std::string cache_dir = header["token_cache_dir"];
      int chunk_index = std::atoi(header["chunk_index"].c_str());
      int chunk_count = std::max(1, std::atoi(header["chunk_count"].c_str()));
      long long total_bytes = std::atoll(header["total_bytes"].c_str());
      if (cache_key.empty() || chunk_index < 0 || chunk_index >= chunk_count || total_bytes < 0)
            throw std::runtime_error("[DIST] Token-cache chunk metadata is invalid."
                                     "\n[ES] La metadata del chunk de cache de tokens no es valida.");

      std::string final_path = token_cache_path_for_worker(cache_dir, cache_key);
      size_t slash = final_path.find_last_of("/\\");
      if (slash != std::string::npos && !ensure_dir(final_path.substr(0, slash)))
            throw std::runtime_error("[DIST] Cannot create worker token-cache directory."
                                     "\n[ES] No se puede crear la carpeta cache de tokens del worker.");

      std::string part_path = final_path + ".part";
      const char *mode = (chunk_index == 0) ? "wb" : "ab";
      FILE *f = std::fopen(part_path.c_str(), mode);
      if (!f)
            throw std::runtime_error("[DIST] Cannot write shared token cache."
                                     "\n[ES] No se puede escribir el cache de tokens compartido.");

      const std::string chunk = payload.substr(body_offset);
      if (!chunk.empty() && std::fwrite(chunk.data(), 1, chunk.size(), f) != chunk.size())
      {
            std::fclose(f);
            throw std::runtime_error("[DIST] Token-cache write failed."
                                     "\n[ES] Fallo la escritura del cache de tokens.");
      }
      std::fclose(f);

      if (chunk_index + 1 == chunk_count)
      {
            if (std::rename(part_path.c_str(), final_path.c_str()) != 0)
                  throw std::runtime_error("[DIST] Cannot finalize shared token cache."
                                           "\n[ES] No se puede finalizar el cache de tokens compartido.");
      }

      std::ostringstream out;
      out << "{\"ok\":true,\"chunk\":" << chunk_index
          << ",\"chunks\":" << chunk_count
          << ",\"bytes\":" << chunk.size()
          << ",\"totalBytes\":" << total_bytes
          << ",\"path\":\"" << json_escape(final_path) << "\"}";
      return out.str();
}

inline std::string handle_token_cache_check_payload(const std::string &payload)
{
      size_t body_offset = 0;
      std::map<std::string, std::string> header = parse_header_payload(payload, body_offset);
      if (header.empty())
            throw std::runtime_error("[DIST] Token-cache check payload is missing headers."
                                     "\n[ES] Al payload de revision de cache de tokens le faltan cabeceras.");

      std::string cache_key = header["token_cache_key"];
      std::string cache_dir = header["token_cache_dir"];
      long long expected_bytes = std::atoll(header["total_bytes"].c_str());
      if (cache_key.empty())
            throw std::runtime_error("[DIST] Token-cache check metadata is invalid."
                                     "\n[ES] La metadata de revision de cache de tokens no es valida.");

      std::string final_path = token_cache_path_for_worker(cache_dir, cache_key);
      struct stat st;
      bool readable = stat(final_path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
      long long bytes = readable ? (long long)st.st_size : -1;
      bool matches = readable && (expected_bytes < 0 || bytes == expected_bytes);

      std::ostringstream out;
      out << "{\"ok\":true,\"exists\":" << (matches ? "true" : "false")
          << ",\"bytes\":" << bytes
          << ",\"expectedBytes\":" << expected_bytes
          << ",\"path\":\"" << json_escape(final_path) << "\"}";
      return out.str();
}

inline bool rpc_request(const std::string &worker,
                        uint16_t type,
                        uint32_t request_id,
                        const std::string &payload,
                        Frame &response,
                        std::string &error,
                        int io_timeout_sec = 900,
                        int connect_timeout_ms = 3000)
{
      std::string host;
      int port = 0;
      if (!parse_host_port(worker, host, port))
      {
            error = "[DIST] Bad worker address / Direccion worker no valida: " + worker;
            return false;
      }

      int fd = connect_to(host, port, connect_timeout_ms);
      if (fd < 0)
      {
            error = "[DIST] Worker unreachable / Worker inaccesible: " + worker;
            return false;
      }

      set_socket_io_timeout(fd, io_timeout_sec);
      bool ok = send_frame(fd, Frame{type, request_id, payload}) && recv_frame(fd, response);
      close(fd);
      if (!ok)
      {
            error = "[DIST] Worker RPC failed / RPC worker fallo: " + worker;
            return false;
      }
      if (response.type == Error)
      {
            error = response.payload;
            return false;
      }
      return true;
}
} // namespace quadtrix_dist_detail

inline int run_distributed_worker(const DistributedConfig &cfg,
                                  int threads,
                                  const std::function<bool()> &should_stop,
                                  const std::function<std::string(const std::string &)> &train_handler =
                                      std::function<std::string(const std::string &)>(),
                                  const std::function<std::string(const std::string &)> &tokenize_handler =
                                      std::function<std::string(const std::string &)>())
{
      using namespace quadtrix_dist_detail;
      int server_fd = socket(AF_INET, SOCK_STREAM, 0);
      if (server_fd < 0)
            throw std::runtime_error("[DIST] Cannot create worker socket.\n[ES] No se puede crear el socket worker.");
      int opt = 1;
      setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      sockaddr_in addr;
      std::memset(&addr, 0, sizeof(addr));
      addr.sin_family = AF_INET;
      addr.sin_port = htons((uint16_t)cfg.worker_port);
      if (inet_pton(AF_INET, cfg.worker_host.c_str(), &addr.sin_addr) != 1)
      {
            close(server_fd);
            throw std::runtime_error("[DIST] Invalid worker host.\n[ES] Host worker no valido.");
      }
      if (bind(server_fd, (sockaddr *)&addr, sizeof(addr)) < 0)
      {
            std::string err = std::strerror(errno);
            close(server_fd);
            throw std::runtime_error("[DIST] Cannot bind worker: " + err +
                                     "\n[ES] No se puede enlazar el worker.");
      }
      if (listen(server_fd, 16) < 0)
      {
            close(server_fd);
            throw std::runtime_error("[DIST] Cannot listen on worker socket.\n[ES] No se puede escuchar en el socket worker.");
      }

      std::cout << "[DIST] Worker listening on " << cfg.worker_host << ":" << cfg.worker_port << "\n";
      std::cout << "[ES] Worker escuchando en " << cfg.worker_host << ":" << cfg.worker_port << "\n";
      std::cout << "[DIST] Trusted LAN token mode, no TLS in v1.\n";
      std::cout << "[ES] Modo token en LAN confiable, sin TLS en v1.\n";

      while (!should_stop())
      {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(server_fd, &rfds);
            timeval tv;
            tv.tv_sec = 1;
            tv.tv_usec = 0;
            int ready = select(server_fd + 1, &rfds, nullptr, nullptr, &tv);
            if (ready <= 0) continue;
            int client = accept(server_fd, nullptr, nullptr);
            if (client < 0) continue;
            std::thread([client, cfg, threads, train_handler, tokenize_handler]() {
                  Frame frame;
                  if (!recv_frame(client, frame))
                  {
                        close(client);
                        return;
                  }
                  std::string token = frame.payload;
                  if (frame.type == TrainStep || frame.type == DatasetChunk ||
                      frame.type == DatasetCheck || frame.type == TokenizeJob ||
                      frame.type == TokenCacheChunk || frame.type == TokenCacheCheck)
                  {
                        size_t body_offset = 0;
                        std::map<std::string, std::string> header = parse_header_payload(frame.payload, body_offset);
                        auto it = header.find("token");
                        token = (it == header.end()) ? "" : it->second;
                  }

                  if (token != cfg.worker_token)
                  {
                        send_frame(client, Frame{Error, frame.request_id, "Bad token / Token incorrecto"});
                        close(client);
                        return;
                  }
                  if (frame.type == Hello || frame.type == Status)
                  {
                        std::cout << "[DIST] Worker status request / Solicitud de estado worker req="
                                  << frame.request_id << "\n";
                        std::cout.flush();
                        std::ostringstream s;
                        s << "{\"ok\":true,\"role\":\"worker\",\"threads\":" << threads
                          << ",\"message\":\"ready / listo\"}";
                        send_frame(client, Frame{Status, frame.request_id, s.str()});
                  }
                  else if (frame.type == Stop)
                  {
                        std::cout << "[DIST] Worker stop request / Solicitud de parada worker req="
                                  << frame.request_id << "\n";
                        std::cout.flush();
                        send_frame(client, Frame{Status, frame.request_id, "{\"ok\":true,\"message\":\"stop noted / parada recibida\"}"});
                  }
                  else if (frame.type == DatasetChunk)
                  {
                        try
                        {
                              size_t body_offset = 0;
                              std::map<std::string, std::string> header = parse_header_payload(frame.payload, body_offset);
                              std::cout << "[DIST] Worker receiving dataset chunk "
                                        << header["chunk_index"] << "/" << header["chunk_count"]
                                        << " bytes=" << (frame.payload.size() >= body_offset ? frame.payload.size() - body_offset : 0)
                                        << " path=" << header["data_path"] << "\n";
                              std::cout << "[ES] Worker recibiendo chunk de dataset "
                                        << header["chunk_index"] << "/" << header["chunk_count"]
                                        << " bytes=" << (frame.payload.size() >= body_offset ? frame.payload.size() - body_offset : 0)
                                        << " ruta=" << header["data_path"] << "\n";
                              std::cout.flush();
                              std::string ack = handle_dataset_chunk_payload(frame.payload);
                              send_frame(client, Frame{DatasetAck, frame.request_id, ack});
                        }
                        catch (const std::exception &e)
                        {
                              send_frame(client, Frame{Error, frame.request_id, e.what()});
                        }
                  }
                  else if (frame.type == DatasetCheck)
                  {
                        try
                        {
                              size_t body_offset = 0;
                              std::map<std::string, std::string> header = parse_header_payload(frame.payload, body_offset);
                              std::cout << "[DIST] Worker dataset cache check req=" << frame.request_id
                                        << " path=" << header["data_path"] << "\n";
                              std::cout << "[ES] Worker revisa cache de dataset req=" << frame.request_id
                                        << " ruta=" << header["data_path"] << "\n";
                              std::cout.flush();
                              std::string status = handle_dataset_check_payload(frame.payload);
                              send_frame(client, Frame{DatasetStatus, frame.request_id, status});
                        }
                        catch (const std::exception &e)
                        {
                              send_frame(client, Frame{Error, frame.request_id, e.what()});
                        }
                  }
                  else if (frame.type == TokenCacheChunk)
                  {
                        try
                        {
                              size_t body_offset = 0;
                              std::map<std::string, std::string> header = parse_header_payload(frame.payload, body_offset);
                              std::cout << "[DIST] Worker receiving Qwen token-cache chunk "
                                        << header["chunk_index"] << "/" << header["chunk_count"]
                                        << " bytes=" << (frame.payload.size() >= body_offset ? frame.payload.size() - body_offset : 0)
                                        << " key=" << header["token_cache_key"] << "\n";
                              std::cout << "[ES] Worker recibiendo chunk de cache de tokens Qwen "
                                        << header["chunk_index"] << "/" << header["chunk_count"]
                                        << " bytes=" << (frame.payload.size() >= body_offset ? frame.payload.size() - body_offset : 0)
                                        << " clave=" << header["token_cache_key"] << "\n";
                              std::cout.flush();
                              std::string ack = handle_token_cache_chunk_payload(frame.payload);
                              send_frame(client, Frame{TokenCacheAck, frame.request_id, ack});
                        }
                        catch (const std::exception &e)
                        {
                              send_frame(client, Frame{Error, frame.request_id, e.what()});
                        }
                  }
                  else if (frame.type == TokenCacheCheck)
                  {
                        try
                        {
                              size_t body_offset = 0;
                              std::map<std::string, std::string> header = parse_header_payload(frame.payload, body_offset);
                              std::cout << "[DIST] Worker Qwen token-cache check req=" << frame.request_id
                                        << " key=" << header["token_cache_key"] << "\n";
                              std::cout << "[ES] Worker revisa cache de tokens Qwen req=" << frame.request_id
                                        << " clave=" << header["token_cache_key"] << "\n";
                              std::cout.flush();
                              std::string status = handle_token_cache_check_payload(frame.payload);
                              send_frame(client, Frame{TokenCacheStatus, frame.request_id, status});
                        }
                        catch (const std::exception &e)
                        {
                              send_frame(client, Frame{Error, frame.request_id, e.what()});
                        }
                  }
	                  else if (frame.type == TrainStep)
	                  {
	                        size_t body_offset = 0;
	                        std::map<std::string, std::string> header = parse_header_payload(frame.payload, body_offset);
                        std::cout << "[DIST] Worker train step req=" << frame.request_id
                                  << " iter=" << header["iter"]
                                  << " micro_steps=" << header["micro_steps"]
                                  << " batch=" << header["batch_size"]
                                  << " block=" << header["block_size"]
                                  << "\n";
                        std::cout << "[ES] Worker paso de entrenamiento req=" << frame.request_id
                                  << " iter=" << header["iter"]
                                  << " microsteps=" << header["micro_steps"]
                                  << " lote=" << header["batch_size"]
                                  << " bloque=" << header["block_size"]
                                  << "\n";
                        std::cout.flush();
	                        if (!train_handler)
	                        {
	                              send_frame(client, Frame{Error, frame.request_id, "Train handler unavailable / Manejador de entrenamiento no disponible"});
	                        }
	                        else
	                        {
	                              try
	                              {
	                                    std::string result = train_handler(frame.payload);
	                                    send_frame(client, Frame{TrainResult, frame.request_id, result});
	                              }
	                              catch (const std::exception &e)
	                              {
	                                    send_frame(client, Frame{Error, frame.request_id, e.what()});
	                              }
	                        }
	                  }
	                  else if (frame.type == TokenizeJob)
	                  {
	                        size_t body_offset = 0;
	                        std::map<std::string, std::string> header = parse_header_payload(frame.payload, body_offset);
	                        std::cout << "[DIST] Worker tokenize job req=" << frame.request_id
	                                  << " job=" << header["job_id"]
	                                  << " records=" << header["record_count"]
	                                  << " chars=" << header["chars"] << "\n";
	                        std::cout << "[ES] Worker trabajo de tokenizacion req=" << frame.request_id
	                                  << " job=" << header["job_id"]
	                                  << " registros=" << header["record_count"]
	                                  << " caracteres=" << header["chars"] << "\n";
	                        std::cout.flush();
	                        if (!tokenize_handler)
	                        {
	                              send_frame(client, Frame{Error, frame.request_id, "Tokenizer handler unavailable / Manejador de tokenizer no disponible"});
	                        }
	                        else
	                        {
	                              try
	                              {
	                                    std::string result = tokenize_handler(frame.payload);
	                                    send_frame(client, Frame{TokenizeResult, frame.request_id, result});
	                              }
	                              catch (const std::exception &e)
	                              {
	                                    send_frame(client, Frame{Error, frame.request_id, e.what()});
	                              }
	                        }
	                  }
                  else
                  {
                        send_frame(client, Frame{Error, frame.request_id, "Unsupported message / Mensaje no soportado"});
                  }
                  close(client);
            }).detach();
      }
      close(server_fd);
      std::cout << "[DIST] Worker stopped.\n[ES] Worker detenido.\n";
      return 0;
}

inline std::string probe_distributed_workers(const DistributedConfig &cfg)
{
      using namespace quadtrix_dist_detail;
      std::ostringstream out;
      std::vector<std::string> workers = split_workers(cfg.workers);
      out << "[DIST] Probing " << workers.size() << " worker(s).\n";
      out << "[ES] Probando " << workers.size() << " worker(s).\n";
      uint32_t req = 1;
      for (const std::string &worker : workers)
      {
            Frame response;
            std::string error;
            if (rpc_request(worker, Hello, req++, cfg.worker_token, response, error, 3, 2000))
                  out << "[DIST] " << worker << " -> " << response.payload << "\n";
            else
                  out << "[DIST] " << worker << " -> " << error << "\n";
      }
      return out.str();
}
