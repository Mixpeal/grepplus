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

path_export_line() {
  if [[ "$INSTALL_DIR" == "${HOME}/.local/bin" ]]; then
    printf 'export PATH="$HOME/.local/bin:$PATH"'
  else
    printf 'export PATH="%s:$PATH"' "$INSTALL_DIR"
  fi
}

detect_shell_rc() {
  if [[ -n "${ZSH_VERSION:-}" ]] || [[ "${SHELL:-}" == *zsh* ]]; then
    echo "${HOME}/.zshrc"
  elif [[ -f "${HOME}/.bashrc" ]]; then
    echo "${HOME}/.bashrc"
  elif [[ -f "${HOME}/.profile" ]]; then
    echo "${HOME}/.profile"
  fi
}

configure_path() {
  case ":$PATH:" in
    *":$INSTALL_DIR:"*)
      info "grepplus is on your PATH."
      return
      ;;
  esac

  local line rc
  line="$(path_export_line)"
  export PATH="${INSTALL_DIR}:${PATH}"
  info "Added ${INSTALL_DIR} to PATH for this session."

  rc="$(detect_shell_rc)"
  if [[ -z "$rc" ]]; then
    warn "Could not detect a shell profile. Add this line manually:"
    warn "  ${line}"
    return
  fi

  if [[ -f "$rc" ]] && grep -Fq ".local/bin" "$rc" 2>/dev/null && [[ "$INSTALL_DIR" == "${HOME}/.local/bin" ]]; then
    info "PATH entry already in ${rc} — run: source ${rc}"
    return
  fi
  if [[ -f "$rc" ]] && grep -Fq "$INSTALL_DIR" "$rc" 2>/dev/null; then
    info "PATH entry already in ${rc} — run: source ${rc}"
    return
  fi

  if [[ ! -t 0 ]] || [[ "${GREPPLUS_NO_PATH_PROMPT:-}" == 1 ]]; then
    warn "To use grepplus in new terminals, add this line to ${rc}:"
    warn "  ${line}"
    warn "Then run: source ${rc}"
    return
  fi

  printf '\n%s is not on your PATH in new terminals.\n' "$INSTALL_DIR"
  printf 'Add it to %s now? [Y/n] ' "$rc"
  read -r reply
  case "${reply:-Y}" in
    [Yy]|"")
      {
        echo ""
        echo "# grepplus"
        echo "$line"
      } >>"$rc"
      info "Updated ${rc}"
      info "Run: source ${rc}"
      info "Or open a new terminal, then: grepplus --help"
      ;;
    *)
      warn "Skipped. Add manually to ${rc}:"
      warn "  ${line}"
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
  local target tag asset url
  target="$(detect_target)"
  tag="$(latest_tag)"
  [[ -n "$tag" ]] || return 1

  asset="grepplus-${target}.tar.gz"
  url="https://github.com/${REPO}/releases/download/${tag}/${asset}"

  info "Downloading ${tag} (${target})"

  local tmp
  tmp="$(mktemp -d)"
  # RETURN + local tmp trips `set -u` when the trap runs; use EXIT in a subshell.
  (
    trap 'rm -rf "$tmp"' EXIT
    if ! curl -fsSL --retry 3 --retry-delay 2 "$url" -o "${tmp}/${asset}"; then
      warn "no release binary at ${url}"
      exit 1
    fi

    ensure_install_dir
    tar xzf "${tmp}/${asset}" -C "$tmp"
    install -m 755 "${tmp}/grepplus" "${tmp}/gp" "$INSTALL_DIR/"
  ) || return 1

  info "Installed grepplus and gp to ${INSTALL_DIR}"
  configure_path
}

install_with_brew() {
  command -v brew >/dev/null 2>&1 || return 1
  info "Installing with Homebrew (builds from source; may pull rust/llvm)"
  export HOMEBREW_NO_AUTO_UPDATE=1
  export HOMEBREW_NO_ENV_HINTS=1
  export NONINTERACTIVE=1
  brew tap mixpeal/grepplus
  brew trust mixpeal/grepplus 2>/dev/null || true
  if CI=1 brew install grepplus; then
    return 0
  fi
  warn "stable install failed; trying --HEAD"
  CI=1 brew install --HEAD mixpeal/grepplus/grepplus
}

install_with_cargo() {
  command -v cargo >/dev/null 2>&1 || return 1
  info "Installing with cargo (this may take a few minutes)"
  local root="${INSTALL_DIR%/bin}"
  [[ "$root" == "$INSTALL_DIR" ]] && root="${HOME}/.local"
  mkdir -p "${root}/bin"
  cargo install --git "https://github.com/${REPO}.git" --locked --root "$root" gp-cli
  info "Installed with cargo to ${root}/bin"
  configure_path
}

main() {
  case "$METHOD" in
    auto)
      install_from_release || install_with_cargo || install_with_brew || \
        die "install failed — install Rust (https://rustup.rs) or use: brew tap mixpeal/grepplus && brew install grepplus"
      ;;
    release) install_from_release || die "release install failed" ;;
    brew) install_with_brew || die "brew install failed" ;;
    cargo) install_with_cargo || die "cargo install failed" ;;
    *) die "unknown GREPPLUS_INSTALL_METHOD: $METHOD (use auto, release, brew, or cargo)" ;;
  esac
}

main "$@"
