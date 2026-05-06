#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CXX_BIN="${QUADTRIX_CXX:-}"
if [[ -z "$CXX_BIN" ]]; then
  if command -v clang++ >/dev/null 2>&1; then
    CXX_BIN="clang++"
  elif command -v g++ >/dev/null 2>&1; then
    CXX_BIN="g++"
  elif command -v c++ >/dev/null 2>&1; then
    CXX_BIN="c++"
  else
    echo "[ERROR] No C++ compiler found. Install clang++ or g++."
    echo "[ES] No se encontro compilador C++. Instala clang++ o g++."
    exit 1
  fi
fi

OUT="${QUADTRIX_OUT:-quadtrix}"
BUILD_DIR="${QUADTRIX_BUILD_DIR:-build/native}"
mkdir -p "$BUILD_DIR"

ARCH="$(uname -m 2>/dev/null || echo unknown)"
CPUINFO="$BUILD_DIR/cpuinfo.txt"
if [[ -r /proc/cpuinfo ]]; then
  rm -f "$CPUINFO" 2>/dev/null || true
  if ! cat /proc/cpuinfo > "$CPUINFO" 2>/dev/null; then
    CPUINFO="/proc/cpuinfo"
  fi
else
  : > "$CPUINFO"
fi

lc_cpuinfo="$(tr '[:upper:]' '[:lower:]' < "$CPUINFO")"

supports_flag() {
  local flag="$1"
  local test_src="$BUILD_DIR/flag_test.cpp"
  local test_obj="$BUILD_DIR/flag_test.o"
  printf 'int main(){return 0;}\n' > "$test_src"
  "$CXX_BIN" -std=c++17 "$flag" -c "$test_src" -o "$test_obj" >/dev/null 2>&1
}

add_if_supported() {
  local flag="$1"
  if supports_flag "$flag"; then
    CXXFLAGS+=("$flag")
    ACCEPTED_FLAGS+=("$flag")
  else
    REJECTED_FLAGS+=("$flag")
  fi
}

CXXFLAGS=(-std=c++17 -O3 -DNDEBUG -I. -Iinclude)
LDFLAGS=()
ACCEPTED_FLAGS=()
REJECTED_FLAGS=()
NATIVE_CPU="${QUADTRIX_NATIVE_CPU:-0}"

if [[ "${QUADTRIX_KEEP_RTTI:-0}" != "1" ]]; then
  add_if_supported "-fno-rtti"
fi
add_if_supported "-ffunction-sections"
add_if_supported "-fdata-sections"
add_if_supported "-fomit-frame-pointer"
add_if_supported "-fstrict-aliasing"

if printf 'int main(){return 0;}\n' > "$BUILD_DIR/openmp_test.cpp" &&
   "$CXX_BIN" -std=c++17 -fopenmp "$BUILD_DIR/openmp_test.cpp" -o "$BUILD_DIR/openmp_test" >/dev/null 2>&1; then
  CXXFLAGS+=("-fopenmp")
  LDFLAGS+=("-fopenmp")
  ACCEPTED_FLAGS+=("-fopenmp")
else
  REJECTED_FLAGS+=("-fopenmp")
fi

# LTO is useful on phones, but some Android/Termux setups do not ship the
# matching linker plugin. Only keep it when both compile and link succeed.
if supports_flag "-flto"; then
  if printf 'int main(){return 0;}\n' > "$BUILD_DIR/lto_test.cpp" &&
     "$CXX_BIN" -std=c++17 -O2 -flto "$BUILD_DIR/lto_test.cpp" -o "$BUILD_DIR/lto_test" >/dev/null 2>&1; then
    CXXFLAGS+=("-flto")
    LDFLAGS+=("-flto")
    ACCEPTED_FLAGS+=("-flto")
  else
    REJECTED_FLAGS+=("-flto")
  fi
fi

case "$ARCH" in
  aarch64|arm64)
    if [[ "$NATIVE_CPU" == "1" ]] && supports_flag "-mcpu=native"; then
      CXXFLAGS+=("-mcpu=native")
      ACCEPTED_FLAGS+=("-mcpu=native")
    else
      add_if_supported "-march=armv8-a"
      add_if_supported "-mtune=native"
      if [[ "$NATIVE_CPU" != "1" ]]; then
        REJECTED_FLAGS+=("-mcpu=native disabled; set QUADTRIX_NATIVE_CPU=1 / -mcpu=native desactivado; usa QUADTRIX_NATIVE_CPU=1")
      fi
    fi
    ;;
  armv7l|armv8l|arm)
    add_if_supported "-march=armv7-a"
    add_if_supported "-mtune=native"
    if [[ "$NATIVE_CPU" == "1" ]]; then
      add_if_supported "-mfpu=neon"
      add_if_supported "-mfloat-abi=softfp"
    else
      REJECTED_FLAGS+=("ARMv7 NEON codegen disabled; set QUADTRIX_NATIVE_CPU=1 / NEON ARMv7 desactivado; usa QUADTRIX_NATIVE_CPU=1")
    fi
    ;;
  x86_64|amd64)
    if [[ "$NATIVE_CPU" == "1" ]]; then
      add_if_supported "-march=native"
    else
      REJECTED_FLAGS+=("-march=native disabled; set QUADTRIX_NATIVE_CPU=1 / -march=native desactivado; usa QUADTRIX_NATIVE_CPU=1")
    fi
    add_if_supported "-mtune=native"
    ;;
  i386|i686)
    if [[ "$NATIVE_CPU" == "1" ]]; then
      add_if_supported "-march=native"
    else
      REJECTED_FLAGS+=("-march=native disabled; set QUADTRIX_NATIVE_CPU=1 / -march=native desactivado; usa QUADTRIX_NATIVE_CPU=1")
    fi
    add_if_supported "-mtune=native"
    ;;
  *)
    if [[ "$NATIVE_CPU" == "1" ]]; then
      add_if_supported "-march=native"
    else
      REJECTED_FLAGS+=("-march=native disabled; set QUADTRIX_NATIVE_CPU=1 / -march=native desactivado; usa QUADTRIX_NATIVE_CPU=1")
    fi
    ;;
esac

add_if_supported "-pthread"
if printf 'int main(){return 0;}\n' > "$BUILD_DIR/gc_sections_test.cpp" &&
   "$CXX_BIN" -std=c++17 "$BUILD_DIR/gc_sections_test.cpp" -Wl,--gc-sections -o "$BUILD_DIR/gc_sections_test" >/dev/null 2>&1; then
  LDFLAGS+=("-Wl,--gc-sections")
  ACCEPTED_FLAGS+=("-Wl,--gc-sections")
else
  REJECTED_FLAGS+=("-Wl,--gc-sections")
fi

if [[ "${QUADTRIX_FAST_MATH:-0}" == "1" ]]; then
  add_if_supported "-ffast-math"
fi

BLAS_MODE="${QUADTRIX_BLAS:-auto}"
if [[ "$BLAS_MODE" != "0" && "$BLAS_MODE" != "off" ]]; then
  blas_src="$BUILD_DIR/blas_test.cpp"
  cat > "$blas_src" <<'EOF'
#include <cblas.h>
int main(){
  float a[1]={1.0f}, b[1]={2.0f}, c[1]={0.0f};
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1.0f, a, 1, b, 1, 0.0f, c, 1);
  return c[0] > 0.0f ? 0 : 1;
}
EOF
  for lib in openblas blas; do
    if "$CXX_BIN" -std=c++17 "$blas_src" -o "$BUILD_DIR/blas_test_$lib" "-l$lib" >/dev/null 2>&1; then
      CXXFLAGS+=("-DQUADTRIX_USE_BLAS")
      LDFLAGS+=("-l$lib")
      ACCEPTED_FLAGS+=("-DQUADTRIX_USE_BLAS" "-l$lib")
      break
    fi
  done
  if [[ ! " ${CXXFLAGS[*]} " =~ " -DQUADTRIX_USE_BLAS " ]]; then
    REJECTED_FLAGS+=("BLAS/OpenBLAS")
    if [[ "$BLAS_MODE" == "required" || "$BLAS_MODE" == "1" || "$BLAS_MODE" == "on" ]]; then
      echo "[ERROR] QUADTRIX_BLAS requires cblas/OpenBLAS, but it was not found."
      echo "[ES] QUADTRIX_BLAS requiere cblas/OpenBLAS, pero no se encontro."
      exit 1
    fi
  fi
fi

{
  echo "Compiler / Compilador: $CXX_BIN"
  echo "Architecture / Arquitectura: $ARCH"
  echo "Native CPU codegen / Codigo CPU nativo: $NATIVE_CPU"
  echo "Output / Salida: $OUT"
  echo "Accepted flags / Flags aceptados:"
  printf '  %s\n' "${ACCEPTED_FLAGS[@]:-none}"
  echo "Rejected flags / Flags rechazados:"
  printf '  %s\n' "${REJECTED_FLAGS[@]:-none}"
  echo "Full compile command / Comando completo:"
  printf '  %q' "$CXX_BIN" "${CXXFLAGS[@]}" main.cpp -o "$OUT" "${LDFLAGS[@]}"
  printf '\n'
} | tee "$BUILD_DIR/build_flags.log"

"$CXX_BIN" "${CXXFLAGS[@]}" main.cpp -o "$OUT" "${LDFLAGS[@]}"

echo "[OK] Built $OUT"
echo "[ES] Compilado $OUT"
