{
  description = "idris-ml build toolchain — pinned dev + CI shell";

  # Pinned so local development and CI resolve byte-identical tools (this is
  # what kills the clang-tidy version skew). Bump deliberately, in lockstep with
  # any nix-darwin/home-manager config that consumes this flake's toolchain.
  inputs.nixpkgs.url = "github:nixos/nixpkgs/567a49d1913ce81ac6e9582e3553dd90a955875f";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux" # CI (GitHub Actions ubuntu)
        "aarch64-darwin" # local development (Apple Silicon)
        "aarch64-linux"
        "x86_64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));

      # Single source of truth for the full build/test toolchain. nix provides
      # everything AROUND idris2; pack (here) builds & owns the one idris2 from
      # the collection commit pinned in pack.toml — so there is deliberately NO
      # nixpkgs `idris2` (two compilers were the whole problem this removes).
      # The dotfiles consume this exact function via `inputs.idris-ml.lib`.
      toolchainPackages =
        pkgs:
        with pkgs;
        [
          chez # Scheme backend; pack builds idris2 with it (pack.toml scheme = "scheme")
          idris2Packages.pack
          clang-tools # clang-format / clang-tidy — pinned == CI == dev (kills version skew)
          cppcheck
          criterion # C unit-test framework (make test-unit-c-*)
          criterion.dev # headers + pkg-config (separate nixpkgs output; needed to COMPILE the C tests)
          uv
          gnumake
          coreutils
          git
          pkg-config # resolves criterion/openblas include+lib paths for the
          # coverage lane's raw clang (which bypasses the nix cc wrapper's
          # NIX_CFLAGS_COMPILE, so it can't see buildInputs implicitly)
          llvm # llvm-profdata / llvm-cov for `make test-coverage-backend-*`
          # (clang-tools ships only clang-format/clang-tidy; macOS Command
          # Line Tools ship no coverage tools, so the lane needs these here)
        ]
        ++ lib.optionals stdenv.isLinux [
          openblas # cblas.h on Linux; macOS uses the Accelerate framework
        ];

      # The lint lane's toolchain: C-lint tools + the Python it needs to run
      # the structural gates and Python-lint gates fully under nix (no
      # apt/brew/curl, no runner-python skew). Deliberately no idris2/pack, so
      # the lint lane never triggers a compiler build.
      #
      # python3 is here even though `defaultPackages` keeps Python OUT (see the
      # rationale there): the two cases differ. defaultPackages' Python would be
      # a *runtime* interpreter for torch/transformers, which MUST be the
      # uv-pinned one — a second nixpkgs interpreter would diverge. The lint
      # lane's python3 is *infrastructure*: it runs the stdlib-only structural
      # gates (gen-ci-workflow / gen-rename-headers / executor-drift checks) and
      # bootstraps the jupyter venv (`python3 -m venv`). For those a pinned
      # nixpkgs python3 is strictly better than the runner's floating
      # /usr/bin/python3 — same "CI == local, no skew" goal as clang-tidy.
      # The torch-dependent lint gates (ruff/vulture/pyright over
      # torch-importing code) live in lint-full and get their interpreter from
      # the uv venv, not from this python3.
      lintPackages =
        pkgs:
        with pkgs;
        [
          clang-tools
          cppcheck
          gnumake # CI runs `nix develop .#lint --command make lint-c …`
          python3 # structural gates (stdlib) + jupyter venv bootstrap
          uv # ruff/vulture/pyright/pytest dev venv (replaces curl-installed uv)
        ]
        ++ lib.optionals stdenv.isLinux [ openblas ];

      # A no-op `stdbuf` (prepended to PATH by the default devShell's
      # shellHook on macOS — see below). pack wraps its build commands in
      # `stdbuf -oL` for line-buffered output, but on the macOS CI runner
      # nix coreutils' libstdbuf.so is arm64 while the chez-built idris2 is
      # arm64e, so the runner's strict dyld aborts EVERY pack-driven idris2
      # build (the cold bootstrap and `pack build …` alike) with
      # "incompatible architecture … need 'arm64e'" → Abort trap: 6.
      # Injecting nothing sidesteps it.
      noopStdbuf =
        pkgs:
        pkgs.writeShellScriptBin "stdbuf" ''
          while [ $# -gt 0 ]; do
            case "$1" in
              -i | -o | -e) shift 2 ;;
              -i* | -o* | -e* | --input=* | --output=* | --error=*) shift ;;
              --)
                shift
                break
                ;;
              *) break ;;
            esac
          done
          exec "$@"
        '';

      # Full build/test shell = the shared toolchain plus git-tools, which the
      # CI lanes need for `git restore-mtime` (the cross-commit TTC-cache mtime
      # reconciliation in setup-idris-ml). Python is intentionally NOT here —
      # it's provided by uv (the pytorch venv / `uv run`), never nixpkgs.
      defaultPackages =
        pkgs:
        toolchainPackages pkgs
        ++ (with pkgs; [
          git-tools # provides git-restore-mtime
        ])
        ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
          # llvm-profdata / llvm-cov for the Linux coverage lane (the
          # default `cc`=gcc has no source-based coverage; the lane forces
          # clang — see COV_CLANG in flake shellHook + mk/tests.mk). Pinned
          # to the same llvmPackages as `clang` below so the tools match the
          # clang that emitted the profraw.
          pkgs.llvm
        ]
        ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
          # The macOS SDK paired with clang-tools' libc++. The full C++
          # clang-tidy (torch/mlx) must parse against THIS sdk, not the host
          # Command Line Tools sdk: nix's libc++ and the host CLT sdk skew
          # (host <sys/resource.h> uint8_t / <math.h> FP_* fail to resolve
          # against nix's `using_if_exists` cstdint). Exported as
          # IDRISML_MACOS_SDKROOT in the shellHook; mk/config.mk feeds it to
          # clang-tidy as -isysroot. clang-tidy only PARSES, so the lint sdk
          # need not match the runner OS (the build still uses Apple clang).
          pkgs.apple-sdk
        ];
    in
    {
      # Reusable, system-agnostic — used by the devShells below and by advanced
      # consumers that want to apply it to their own pkgs.
      lib.toolchainPackages = toolchainPackages;

      # Pre-applied against the flake's PINNED nixpkgs. External consumers (the
      # dotfiles' `local.idris-ml` module) take THIS, not `lib.toolchainPackages
      # <their pkgs>` — applying the function to a consumer's own (floating)
      # nixpkgs would use their tool versions and reintroduce the dev↔CI skew
      # this whole effort removes. `.toolchain` == the `default` devShell's set.
      legacyPackages = forAllSystems (pkgs: {
        toolchain = toolchainPackages pkgs;
        lint = lintPackages pkgs;
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = defaultPackages pkgs;
          # macOS only: prepend the no-op stdbuf so it wins over coreutils'
          # in the dev shell (mkShell's PATH ignores meta.priority, so a
          # shellHook prepend is the reliable shadow). Keeps the no-op out of
          # the toolchain the dotfiles install system-wide.
          shellHook =
            pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
              export PATH="${noopStdbuf pkgs}/bin:$PATH"
              # SDK root for the full C++ clang-tidy (see apple-sdk above +
              # mk/config.mk CLANG_TIDY_EXTRA_CFLAGS). Referenced by absolute
              # path, NOT added to the build's sysroot — the actual compile
              # still uses Apple clang against the host CLT sdk.
              export IDRISML_MACOS_SDKROOT="${pkgs.apple-sdk}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
            ''
            + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              # Linux coverage lane: a glibc-consistent *wrapped* clang,
              # referenced by absolute path (NOT added to PATH, so it doesn't
              # shadow the stdenv `cc`=gcc the rest of the build uses). The
              # coverage target forces clang for -fprofile-instr-generate /
              # -fcoverage-mapping (gcc rejects them); building it with the
              # wrapped clang keeps the test binary on nix glibc + the nix
              # dynamic linker, so it can load nix criterion (whose
              # libanl.so.1 needs nix glibc 2.42) at runtime instead of
              # aborting with "GLIBC_ABI_DT_X86_64_PLT not found".
              export COV_CLANG="${pkgs.clang}/bin/clang"
              export COV_CLANGXX="${pkgs.clang}/bin/clang++"
            '';
        };
        lint = pkgs.mkShell { packages = lintPackages pkgs; };
      });
    };
}
