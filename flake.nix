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
        ]
        ++ lib.optionals stdenv.isLinux [
          openblas # cblas.h on Linux; macOS uses the Accelerate framework
        ];

      # C-lint only: no idris2/pack, so the lint lane never triggers a compiler
      # build. This is all the nix CI pilot wires in (the rest of the lanes move
      # to the `default` shell in a later stage).
      lintPackages =
        pkgs:
        with pkgs;
        [
          clang-tools
          cppcheck
          gnumake # CI runs `nix develop .#lint --command make lint-c …`
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
        ]);
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
          shellHook = pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
            export PATH="${noopStdbuf pkgs}/bin:$PATH"
          '';
        };
        lint = pkgs.mkShell { packages = lintPackages pkgs; };
      });
    };
}
