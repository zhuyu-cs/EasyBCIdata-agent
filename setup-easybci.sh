#!/bin/bash
# ============================================================================
# EasyBCI Agent Setup Script
# ============================================================================
# Quick setup for developers who cloned the repo manually.
# Uses uv for desktop/server setup and Python's stdlib venv + pip on Termux.
#
# Usage:
#   ./setup-easybci.sh
#
# This script:
# 1. Detects desktop/server vs Android/Termux setup path
# 2. Creates a Python 3.11 virtual environment
# 3. Installs the appropriate dependency set for the platform
# 4. Creates .env from template (if not exists), syncs it to ~/.easybci/.env on first install
# 5. Symlinks the 'easybci' CLI command into a user-facing bin dir
# 6. Runs the setup wizard (optional)
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prevent uv from discovering config files (uv.toml, pyproject.toml) from the
# wrong user's home directory when running under sudo -u <user>.  See #21269.
export UV_NO_CONFIG=1

PYTHON_VERSION="3.11"

is_termux() {
    [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux/files/usr"* ]]
}

get_command_link_dir() {
    if is_termux && [ -n "${PREFIX:-}" ]; then
        echo "$PREFIX/bin"
    else
        echo "$HOME/.local/bin"
    fi
}

get_command_link_display_dir() {
    if is_termux && [ -n "${PREFIX:-}" ]; then
        echo '$PREFIX/bin'
    else
        echo '~/.local/bin'
    fi
}

echo ""
echo -e "${CYAN}⚕ EasyBCI Agent Setup${NC}"
echo ""

# ============================================================================
# Install / locate uv
# ============================================================================

echo -e "${CYAN}→${NC} Checking for uv..."

UV_CMD=""
if is_termux; then
    echo -e "${CYAN}→${NC} Termux detected — using Python's stdlib venv + pip instead of uv"
else
    if command -v uv &> /dev/null; then
        UV_CMD="uv"
    elif [ -x "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
    elif [ -x "$HOME/.cargo/bin/uv" ]; then
        UV_CMD="$HOME/.cargo/bin/uv"
    fi

    if [ -n "$UV_CMD" ]; then
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} uv found ($UV_VERSION)"
    else
        echo -e "${CYAN}→${NC} Installing uv..."
        # Capture installer output so a failure shows the user WHY
        # (network, glibc mismatch on old distros, missing curl, disk
        # full, etc.) instead of "✗ Failed to install uv" with zero
        # diagnostic.  Two-stage to avoid `curl | sh` masking curl
        # failures (sh exits 0 on empty stdin under no pipefail).
        _uv_log="$(mktemp 2>/dev/null || echo "/tmp/easybci-uv-install.$$.log")"
        _uv_installer="$(mktemp 2>/dev/null || echo "/tmp/easybci-uv-installer.$$.sh")"
        if ! curl -LsSf https://astral.sh/uv/install.sh -o "$_uv_installer" 2>"$_uv_log"; then
            echo -e "${RED}✗${NC} Failed to download uv installer."
            sed 's/^/    /' "$_uv_log" >&2
            echo -e "${CYAN}→${NC} Install manually: https://docs.astral.sh/uv/"
            rm -f "$_uv_log" "$_uv_installer"
            exit 1
        fi
        if sh "$_uv_installer" >>"$_uv_log" 2>&1; then
            rm -f "$_uv_installer"
            if [ -x "$HOME/.local/bin/uv" ]; then
                UV_CMD="$HOME/.local/bin/uv"
            elif [ -x "$HOME/.cargo/bin/uv" ]; then
                UV_CMD="$HOME/.cargo/bin/uv"
            fi

            if [ -n "$UV_CMD" ]; then
                rm -f "$_uv_log"
                UV_VERSION=$($UV_CMD --version 2>/dev/null)
                echo -e "${GREEN}✓${NC} uv installed ($UV_VERSION)"
            else
                echo -e "${RED}✗${NC} uv installer reported success but binary not found. Add ~/.local/bin to PATH and retry."
                echo -e "${CYAN}→${NC} Installer output:"
                sed 's/^/    /' "$_uv_log" >&2
                rm -f "$_uv_log"
                exit 1
            fi
        else
            echo -e "${RED}✗${NC} Failed to install uv."
            echo -e "${CYAN}→${NC} Installer output:"
            sed 's/^/    /' "$_uv_log" >&2
            echo -e "${CYAN}→${NC} Install manually: https://docs.astral.sh/uv/"
            rm -f "$_uv_log" "$_uv_installer"
            exit 1
        fi
    fi
fi

# ============================================================================
# Python check (uv can provision it automatically)
# ============================================================================

echo -e "${CYAN}→${NC} Checking Python $PYTHON_VERSION..."

if is_termux; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_PATH="$(command -v python)"
        if "$PYTHON_PATH" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
            echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION found"
        else
            echo -e "${RED}✗${NC} Termux Python must be 3.11+"
            echo "    Run: pkg install python"
            exit 1
        fi
    else
        echo -e "${RED}✗${NC} Python not found in Termux"
        echo "    Run: pkg install python"
        exit 1
    fi
else
    if $UV_CMD python find "$PYTHON_VERSION" &> /dev/null; then
        PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
        PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION found"
    else
        echo -e "${CYAN}→${NC} Python $PYTHON_VERSION not found, installing via uv..."
        $UV_CMD python install "$PYTHON_VERSION"
        PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
        PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION installed"
    fi
fi

# ============================================================================
# Virtual environment
# ============================================================================

echo -e "${CYAN}→${NC} Setting up virtual environment..."

if [ -d "venv" ]; then
    echo -e "${CYAN}→${NC} Removing old venv..."
    rm -rf venv
fi

if is_termux; then
    "$PYTHON_PATH" -m venv venv
    echo -e "${GREEN}✓${NC} venv created with stdlib venv"
else
    $UV_CMD venv venv --python "$PYTHON_VERSION"
    echo -e "${GREEN}✓${NC} venv created (Python $PYTHON_VERSION)"
fi

export VIRTUAL_ENV="$SCRIPT_DIR/venv"
SETUP_PYTHON="$SCRIPT_DIR/venv/bin/python"

# ============================================================================
# Dependencies
# ============================================================================

echo -e "${CYAN}→${NC} Installing dependencies..."

if is_termux; then
    export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk 2>/dev/null || printf '%s' "${ANDROID_API_LEVEL:-}")"
    echo -e "${CYAN}→${NC} Termux detected — installing the tested Android bundle"
    "$SETUP_PYTHON" -m pip install --upgrade pip setuptools wheel
    # `.[termux,exa]`: fold in the exa web-search backend (pure-Python SDK,
    # safe on Android) so the agent doesn't lazy-install it mid-session.
    if [ -f "constraints-termux.txt" ]; then
        "$SETUP_PYTHON" -m pip install -e ".[termux,exa]" -c constraints-termux.txt || {
            echo -e "${YELLOW}⚠${NC} Termux bundle install failed, falling back to base install..."
            "$SETUP_PYTHON" -m pip install -e "." -c constraints-termux.txt
        }
    else
        "$SETUP_PYTHON" -m pip install -e ".[termux,exa]" || "$SETUP_PYTHON" -m pip install -e "."
    fi
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    # Prefer uv sync with lockfile (hash-verified installs) when available,
    # fall back to pip install for compatibility or when lockfile is stale.
    #
    # Multi-tier pip fallback. Goal: ONE compromised PyPI package
    # (mistralai 2.4.6 in May 2026 → quarantined) shouldn't silently demote
    # a fresh setup to "core only". Edit _BROKEN_EXTRAS when a transitive
    # breaks; users keep voice / honcho / google / matrix etc. even
    # if mistral can't resolve.
    _BROKEN_EXTRAS=()  # populate when an extra becomes unresolvable
    _ALL_EXTRAS=(
        modal daytona vercel matrix cron cli dev tts-premium
        pty honcho mcp homeassistant sms acp voice dingtalk feishu google
        bedrock web youtube
        # exa: web-search backend, eager-installed at setup so the agent
        # never pays a first-use `pip install exa-py` mid-session. NOT in
        # pyproject's [all] extra (search backends stay lazy there); we add
        # it explicitly here. tavily needs no package (HTTP-only via httpx).
        exa
    )
    _SAFE_EXTRAS=()
    for _e in "${_ALL_EXTRAS[@]}"; do
        _skip=false
        for _b in "${_BROKEN_EXTRAS[@]}"; do
            [ "$_e" = "$_b" ] && _skip=true && break
        done
        [ "$_skip" = false ] && _SAFE_EXTRAS+=("$_e")
    done
    _SAFE_SPEC=".[$(IFS=,; echo "${_SAFE_EXTRAS[*]}")]"
    _try_install() {
        # `.[all,exa]`: eager-install exa-py alongside the curated [all]
        # extra (exa is intentionally kept out of [all] itself — see the
        # pyproject.toml policy comment).
        $UV_CMD pip install -e ".[all,exa]" \
            || $UV_CMD pip install -e "$_SAFE_SPEC" \
            || $UV_CMD pip install -e "."
    }

    if [ -f "uv.lock" ]; then
        # Hash-verified install (preferred). The lockfile records SHA256
        # hashes for every transitive — a compromised transitive would have
        # a different hash and be REJECTED by uv. This is the only path
        # that protects against transitive-package supply-chain attacks
        # (the direct deps in pyproject.toml are exact-pinned, but
        # `uv pip install` re-resolves transitives fresh from PyPI).
        echo -e "${CYAN}→${NC} Using uv.lock for hash-verified installation..."
        echo -e "${CYAN}→${NC} (first run on a fresh venv can take 1-5 minutes; uv prints progress below)"
        # Critical flag choice: `--extra all`, NOT `--all-extras`. The
        # latter installs every [project.optional-dependencies] key,
        # bypassing the curated [all] extra and pulling backends like
        # [matrix] (python-olm needs make on Windows) and [rl] (git+https
        # deps that fail offline). See pyproject.toml's [all] for the
        # curated set, and tools/lazy_deps.py for backends that install
        # at first use.
        # Also: stream stderr through directly so the user sees uv's
        # progress UI instead of staring at a frozen prompt.
        #
        # `--extra exa` is added ON TOP of `--extra all`: the web-search
        # backend exa-py is deliberately NOT in pyproject's [all] extra
        # (search backends stay lazy-installable there — see that policy
        # comment), but we eager-install it here so a running agent never
        # pays a first-use `pip install exa-py` mid-session. exa-py is
        # already in uv.lock, so this stays hash-verified under --locked.
        # tavily needs no package (HTTP-only via httpx, already core).
        if UV_PROJECT_ENVIRONMENT="$SCRIPT_DIR/venv" $UV_CMD sync --extra all --extra exa --locked; then
            echo -e "${GREEN}✓${NC} Dependencies installed (hash-verified via uv.lock)"
        else
            echo -e "${YELLOW}⚠${NC} Lockfile sync failed (see uv output above)."
            echo -e "${YELLOW}⚠${NC} Falling back to PyPI resolve — transitives will NOT be hash-verified."
            _try_install
            echo -e "${GREEN}✓${NC} Dependencies installed (transitives re-resolved, not hash-verified)"
        fi
    else
        echo -e "${YELLOW}⚠${NC} uv.lock not found — installing without hash verification of transitives."
        _try_install
        echo -e "${GREEN}✓${NC} Dependencies installed (transitives re-resolved, not hash-verified)"
    fi
fi

# ============================================================================
# ============================================================================
# Optional: ripgrep (for faster file search)
# ============================================================================

echo -e "${CYAN}→${NC} Checking ripgrep (optional, for faster search)..."

if command -v rg &> /dev/null; then
    echo -e "${GREEN}✓${NC} ripgrep found"
else
    echo -e "${YELLOW}⚠${NC} ripgrep not found (file search will use grep fallback)"
    read -p "Install ripgrep for faster search? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        INSTALLED=false

        if is_termux; then
            pkg install -y ripgrep && INSTALLED=true
        else
            # Check if sudo is available
            if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
                if command -v apt &> /dev/null; then
                    sudo apt install -y ripgrep && INSTALLED=true
                elif command -v dnf &> /dev/null; then
                    sudo dnf install -y ripgrep && INSTALLED=true
                fi
            fi

            # Try brew (no sudo needed)
            if [ "$INSTALLED" = false ] && command -v brew &> /dev/null; then
                brew install ripgrep && INSTALLED=true
            fi

            # Try cargo (no sudo needed)
            if [ "$INSTALLED" = false ] && command -v cargo &> /dev/null; then
                echo -e "${CYAN}→${NC} Trying cargo install (no sudo required)..."
                cargo install ripgrep && INSTALLED=true
            fi
        fi

        if [ "$INSTALLED" = true ]; then
            echo -e "${GREEN}✓${NC} ripgrep installed"
        else
            echo -e "${YELLOW}⚠${NC} Auto-install failed. Install options:"
            if is_termux; then
                echo "    pkg install ripgrep          # Termux / Android"
            else
                echo "    sudo apt install ripgrep     # Debian/Ubuntu"
                echo "    brew install ripgrep         # macOS"
                echo "    cargo install ripgrep        # With Rust (no sudo)"
            fi
            echo "    https://github.com/BurntSushi/ripgrep#installation"
        fi
    fi
fi

# ============================================================================
# Environment file
# ============================================================================

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓${NC} Created .env from template"
    fi
else
    echo -e "${GREEN}✓${NC} .env exists"
fi

# Copy repo .env to ~/.easybci/.env on first install so the user-config
# location (which env_loader.py treats as the authoritative source, see
# easybci_cli/env_loader.py:142) is populated. Keeps the repo .env in place
# as a dev fallback; never overwrites an existing user .env to avoid
# clobbering keys the user has already added via `easybci setup` / doctor.
EASYBCI_HOME_DIR="${EASYBCI_HOME:-$HOME/.easybci}"
USER_ENV_FILE="$EASYBCI_HOME_DIR/.env"
if [ -f ".env" ]; then
    if [ ! -f "$USER_ENV_FILE" ]; then
        mkdir -p "$EASYBCI_HOME_DIR"
        cp .env "$USER_ENV_FILE"
        chmod 600 "$USER_ENV_FILE" 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Synced .env → $USER_ENV_FILE (chmod 600)"
    else
        echo -e "${GREEN}✓${NC} $USER_ENV_FILE already exists — leaving it untouched (repo .env kept as dev fallback)"
    fi
fi

# ============================================================================
# Snapshot main provider creds into auxiliary.* / delegation.* and probe
# the provider's /v1/models for entitlement.  Idempotent — only fills
# blanks, preserves user overrides, runs every install.
# See easybci_cli/sync_config.py for the full contract.
# ============================================================================

if [ -x "$SCRIPT_DIR/venv/bin/python" ] && [ -f "$EASYBCI_HOME_DIR/config.yaml" ]; then
    echo -e "${CYAN}→${NC} Syncing main provider creds into auxiliary slots..."
    EASYBCI_HOME="$EASYBCI_HOME_DIR" "$SCRIPT_DIR/venv/bin/python" -m easybci_cli.sync_config || \
        echo -e "${YELLOW}⚠${NC}  sync_config exited non-zero — install will continue, run \`easybci doctor\` to inspect."
fi

# ============================================================================
# PATH setup — symlink easybci into a user-facing bin dir
# ============================================================================

echo -e "${CYAN}→${NC} Setting up easybci command..."

EASYBCI_BIN="$SCRIPT_DIR/venv/bin/easybci"
COMMAND_LINK_DIR="$(get_command_link_dir)"
COMMAND_LINK_DISPLAY_DIR="$(get_command_link_display_dir)"
mkdir -p "$COMMAND_LINK_DIR"
ln -sf "$EASYBCI_BIN" "$COMMAND_LINK_DIR/easybci"
echo -e "${GREEN}✓${NC} Symlinked easybci → $COMMAND_LINK_DISPLAY_DIR/easybci"

if is_termux; then
    export PATH="$COMMAND_LINK_DIR:$PATH"
    echo -e "${GREEN}✓${NC} $COMMAND_LINK_DISPLAY_DIR is already on PATH in Termux"
else
    # Determine the appropriate shell config file
    SHELL_CONFIG=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        SHELL_CONFIG="$HOME/.bashrc"
        [ ! -f "$SHELL_CONFIG" ] && SHELL_CONFIG="$HOME/.bash_profile"
    else
        # Fallback to checking existing files
        if [ -f "$HOME/.zshrc" ]; then
            SHELL_CONFIG="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_CONFIG="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
            SHELL_CONFIG="$HOME/.bash_profile"
        fi
    fi

    if [ -n "$SHELL_CONFIG" ]; then
        # Touch the file just in case it doesn't exist yet but was selected
        touch "$SHELL_CONFIG" 2>/dev/null || true

        if ! echo "$PATH" | tr ':' '\n' | grep -q "^$HOME/.local/bin$"; then
            if ! grep -q '\.local/bin' "$SHELL_CONFIG" 2>/dev/null; then
                echo "" >> "$SHELL_CONFIG"
                echo "# EasyBCI Agent — ensure ~/.local/bin is on PATH" >> "$SHELL_CONFIG"
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
                echo -e "${GREEN}✓${NC} Added ~/.local/bin to PATH in $SHELL_CONFIG"
            else
                echo -e "${GREEN}✓${NC} ~/.local/bin already in $SHELL_CONFIG"
            fi
        else
            echo -e "${GREEN}✓${NC} ~/.local/bin already on PATH"
        fi
    fi
fi

# ============================================================================
# Seed bundled skills into ~/.easybci/skills/
# ============================================================================

EASYBCI_SKILLS_DIR="${EASYBCI_HOME:-$HOME/.easybci}/skills"
mkdir -p "$EASYBCI_SKILLS_DIR"

echo ""
echo "Syncing bundled skills to ~/.easybci/skills/ ..."
# Run the sync module from the repo root so `-m` resolves the package. Do NOT
# swallow its output — a silent failure here leaves ~/.easybci/skills/ empty.
if (cd "$SCRIPT_DIR" && "$SCRIPT_DIR/venv/bin/python" -m easybci_lib.tools.skills_sync); then
    echo -e "${GREEN}✓${NC} Skills synced"
else
    # Fallback: copy bundled skills verbatim if the sync module fails.
    echo -e "${YELLOW}!${NC} Skill sync failed; falling back to plain copy."
    if [ -d "$SCRIPT_DIR/easybci_lib/skills" ]; then
        cp -rn "$SCRIPT_DIR/easybci_lib/skills/"* "$EASYBCI_SKILLS_DIR/" 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Skills copied"
    else
        echo -e "${RED}✗${NC} Bundled skills dir not found at easybci_lib/skills/"
    fi
fi

# ============================================================================
# Optional: build the WebUI (only if user touched easybci_web/ sources)
# ============================================================================
# The wheel ships a pre-built bundle in easybci_cli/web_dist/, so skip when:
#   • web_dist/ exists AND is newer than easybci_web/src/  → nothing to do
#   • Node 20+ is not installed                            → tell the user how
# Always rebuild when easybci_web/src/ has been edited (developers' workflow).
WEB_SRC_DIR="$SCRIPT_DIR/easybci_web"
WEB_DIST_DIR="$SCRIPT_DIR/easybci_cli/web_dist"

_needs_web_build() {
    [ -d "$WEB_SRC_DIR/src" ] || return 1
    [ -f "$WEB_DIST_DIR/index.html" ] || return 0
    # Rebuild if any src file is newer than the built index.html.
    local newest_src
    newest_src="$(find "$WEB_SRC_DIR/src" "$WEB_SRC_DIR/package.json" \
        "$WEB_SRC_DIR/vite.config.ts" "$WEB_SRC_DIR/index.html" \
        -type f -newer "$WEB_DIST_DIR/index.html" -print -quit 2>/dev/null)"
    [ -n "$newest_src" ]
}

_NODE_MIN_MAJOR=20

_install_node20() {
    # Returns 0 on success, 1 on failure. Tries package managers that satisfy
    # Node $_NODE_MIN_MAJOR+, then falls back to nvm (no-sudo, per-user).
    if is_termux; then
        pkg install -y nodejs && return 0 || return 1
    fi
    # apt (Debian/Ubuntu) via NodeSource
    if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
        if sudo -n true 2>/dev/null; then
            echo -e "${CYAN}→${NC} Installing Node ${_NODE_MIN_MAJOR} via NodeSource (apt)..."
            if curl -fsSL "https://deb.nodesource.com/setup_${_NODE_MIN_MAJOR}.x" | sudo -E bash - >/dev/null 2>&1 \
                && sudo apt-get install -y nodejs >/dev/null 2>&1; then
                return 0
            fi
        else
            echo -e "${CYAN}→${NC} apt requires sudo password — running NodeSource installer interactively..."
            if curl -fsSL "https://deb.nodesource.com/setup_${_NODE_MIN_MAJOR}.x" | sudo -E bash - \
                && sudo apt-get install -y nodejs; then
                return 0
            fi
        fi
    fi
    # dnf (Fedora/RHEL)
    if command -v dnf >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
        echo -e "${CYAN}→${NC} Installing Node ${_NODE_MIN_MAJOR} via dnf..."
        if curl -fsSL "https://rpm.nodesource.com/setup_${_NODE_MIN_MAJOR}.x" | sudo -E bash - \
            && sudo dnf install -y nodejs; then
            return 0
        fi
    fi
    # brew (macOS / Linuxbrew, no sudo)
    if command -v brew >/dev/null 2>&1; then
        echo -e "${CYAN}→${NC} Installing Node via brew..."
        if brew install "node@${_NODE_MIN_MAJOR}"; then
            brew link --overwrite --force "node@${_NODE_MIN_MAJOR}" 2>/dev/null || true
            return 0
        fi
    fi
    # nvm fallback (no sudo, per-user)
    echo -e "${CYAN}→${NC} Falling back to nvm (no sudo, per-user install)..."
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    if [ ! -s "$NVM_DIR/nvm.sh" ]; then
        curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash >/dev/null 2>&1 || return 1
    fi
    # shellcheck source=/dev/null
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
    nvm install "${_NODE_MIN_MAJOR}" >/dev/null 2>&1 || return 1
    nvm use "${_NODE_MIN_MAJOR}" >/dev/null 2>&1 || true
    nvm alias default "${_NODE_MIN_MAJOR}" >/dev/null 2>&1 || true
    # Make sure PATH for THIS script picks up the nvm node.
    export PATH="$NVM_DIR/versions/node/$(nvm version)/bin:$PATH"
    return 0
}

if _needs_web_build; then
    echo ""
    echo -e "${CYAN}→${NC} WebUI sources changed (or no prebuilt bundle found) — building..."
    _node_ok=false
    if command -v node >/dev/null 2>&1; then
        _node_major="$(node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/')"
        if [ -n "$_node_major" ] && [ "$_node_major" -ge "$_NODE_MIN_MAJOR" ]; then
            _node_ok=true
        else
            echo -e "${YELLOW}⚠${NC} Node $_node_major detected — Vite 7 requires Node ${_NODE_MIN_MAJOR}+."
        fi
    else
        echo -e "${YELLOW}⚠${NC} Node.js not found."
    fi

    if [ "$_node_ok" = false ]; then
        # Auto-install Node $_NODE_MIN_MAJOR+ so the WebUI works out of the box.
        # Asks once; honors -y/--yes if the user wants fully unattended setup.
        _do_install_node=false
        if [ "${EASYBCI_AUTO_INSTALL_NODE:-}" = "1" ] || [ "${YES:-}" = "1" ]; then
            _do_install_node=true
        else
            read -p "Install Node ${_NODE_MIN_MAJOR} now so the WebUI builds? [Y/n] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                _do_install_node=true
            fi
        fi
        if [ "$_do_install_node" = true ]; then
            if _install_node20; then
                _node_major="$(node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/')"
                if [ -n "$_node_major" ] && [ "$_node_major" -ge "$_NODE_MIN_MAJOR" ]; then
                    echo -e "${GREEN}✓${NC} Node $(node --version) installed"
                    _node_ok=true
                else
                    echo -e "${YELLOW}⚠${NC} Node install completed but version is still <${_NODE_MIN_MAJOR}. Restart your shell and retry."
                fi
            else
                echo -e "${YELLOW}⚠${NC} Auto-install failed. Install Node ${_NODE_MIN_MAJOR}+ manually then rerun this script:"
                echo "      curl -fsSL https://deb.nodesource.com/setup_${_NODE_MIN_MAJOR}.x | sudo -E bash -"
                echo "      sudo apt-get install -y nodejs   # or: brew install node@${_NODE_MIN_MAJOR}"
            fi
        else
            echo "    Skipping WebUI build — install Node ${_NODE_MIN_MAJOR}+ and rerun later."
        fi
    fi

    if [ "$_node_ok" = true ]; then
        (
            cd "$WEB_SRC_DIR" || exit 1
            if [ ! -d node_modules ]; then
                echo -e "${CYAN}→${NC} Installing npm deps (one-time, ~1 min)..."
                npm install --silent 2>&1 | tail -3
            fi
            if npm run build 2>&1 | tail -3; then
                echo -e "${GREEN}✓${NC} WebUI built into $(realpath --relative-to="$SCRIPT_DIR" "$WEB_DIST_DIR")/"
            else
                echo -e "${YELLOW}⚠${NC} WebUI build failed — dashboard will serve stale bundle if any."
            fi
        )
    fi
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo ""
if is_termux; then
    echo "  1. Run the setup wizard to configure API keys:"
    echo "     easybci setup"
    echo ""
    echo "  2. Start chatting:"
    echo "     easybci"
    echo ""
else
    echo "  1. Reload your shell:"
    echo "     source $SHELL_CONFIG"
    echo ""
    echo "  2. Run the setup wizard to configure API keys:"
    echo "     easybci setup"
    echo ""
    echo "  3. Start chatting:"
    echo "     easybci"
    echo ""
fi
echo "Other commands:"
echo "  easybci status        # Check configuration"
if is_termux; then
    echo "  easybci gateway       # Run gateway in foreground"
else
    echo "  easybci gateway install # Install gateway service (messaging + cron)"
fi
echo "  easybci cron list     # View scheduled jobs"
echo "  easybci doctor        # Diagnose issues"
echo ""

# Ask if they want to run setup wizard now
read -p "Would you like to run the setup wizard now? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo ""
    # Run directly with venv Python (no activation needed)
    "$SCRIPT_DIR/venv/bin/python" -m easybci_cli.main setup
fi
