#!/usr/bin/env bash
# grep+ installer — https://github.com/Mixpeal/grepplus
set -euo pipefail

REPO="Mixpeal/grepplus"
INSTALL_DIR="${GREPPLUS_INSTALL_DIR:-${HOME}/.local/bin}"
METHOD="${GREPPLUS_INSTALL_METHOD:-auto}"

info() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

ensure_install_dir() {
  mkdir -p "$INSTALL_DIR"
}

path_hint() {
  case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
      warn "$INSTALL_DIR is not on your PATH."
      warn "Add: export PATH=\"$INSTALL_DIR:\$PATH\""
      ;;
  esac
}

detect_target() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os:$arch" in
    Darwin:arm64|Darwin:aarch64) echo "aarch64-apple-darwin" ;;
    Darwin:x86_64) echo "x86_64-apple-darwin" ;;
    Linux:x86_64) echo "x86_64-unknown-linux-gnu" ;;
    Linux:aarch64|Linux:arm64) echo "aarch64-unknown-linux-gnu" ;;
    *) die "unsupported platform: $os $arch" ;;
  esac
}

latest_tag() {
  if [[ -n "${GREPPLUS_VERSION:-}" ]]; then
    printf '%s\n' "$GREPPLUS_VERSION"
    return
  fi
  # No -f: /releases/latest returns 404 when no release exists yet.
  curl -sSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -n1
}

install_from_release() {
  local target tag asset url tmp
  target="$(detect_target)"
  tag="$(latest_tag)"
  [[ -n "$tag" ]] || return 1

  asset="grepplus-${target}.tar.gz"
  url="https://github.com/${REPO}/releases/download/${tag}/${asset}"

  info "Downloading ${tag} (${target})"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  if ! curl -fsSL "$url" -o "${tmp}/${asset}"; then
    warn "no release binary at ${url}"
    return 1
  fi

  ensure_install_dir
  tar xzf "${tmp}/${asset}" -C "$tmp"
  install -m 755 "${tmp}/grepplus" "${tmp}/gp" "$INSTALL_DIR/"

  info "Installed grepplus and gp to ${INSTALL_DIR}"
  path_hint
}

install_with_brew() {
  command -v brew >/dev/null 2>&1 || return 1
  info "Installing with Homebrew"
  brew tap mixpeal/grepplus
  brew trust mixpeal/grepplus 2>/dev/null || true
  if brew install grepplus; then
    return 0
  fi
  warn "stable install failed (push tag v0.1.0 for release tarballs); trying --HEAD"
  brew install --HEAD mixpeal/grepplus/grepplus
}

install_with_cargo() {
  command -v cargo >/dev/null 2>&1 || return 1
  info "Installing with cargo (this may take a few minutes)"
  local root="${INSTALL_DIR%/bin}"
  [[ "$root" == "$INSTALL_DIR" ]] && root="${HOME}/.local"
  mkdir -p "${root}/bin"
  cargo install --git "https://github.com/${REPO}.git" --locked --root "$root" gp-cli
  info "Installed with cargo to ${root}/bin"
  path_hint
}

main() {
  case "$METHOD" in
    auto)
      install_from_release || install_with_brew || install_with_cargo || \
        die "install failed — install Rust (https://rustup.rs) or Homebrew (https://brew.sh) and retry"
      ;;
    release) install_from_release || die "release install failed" ;;
    brew) install_with_brew || die "brew install failed" ;;
    cargo) install_with_cargo || die "cargo install failed" ;;
    *) die "unknown GREPPLUS_INSTALL_METHOD: $METHOD (use auto, release, brew, or cargo)" ;;
  esac
}

main "$@"
