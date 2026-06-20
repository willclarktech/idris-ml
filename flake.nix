{
  description = "idris-ml build toolchain — pinned dev + CI shell";

  # Pinned to match the dev box's nix-darwin flake (~/code/dotfiles/vm) so the
  # dev box and CI resolve byte-identical tools. Bump deliberately, in lockstep
  # with the dotfiles, when refreshing the toolchain.
  inputs.nixpkgs.url = "github:nixos/nixpkgs/567a49d1913ce81ac6e9582e3553dd90a955875f";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux" # CI (GitHub Actions ubuntu)
        "aarch64-darwin" # dev box (Apple Silicon)
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
        default = pkgs.mkShell { packages = toolchainPackages pkgs; };
        lint = pkgs.mkShell { packages = lintPackages pkgs; };
      });
    };
}
