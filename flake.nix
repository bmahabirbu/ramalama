{
  inputs = {
    nixpkgs = {
      url = "github:NixOS/nixpkgs/nixos-unstable";
    };
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

      mkRamalama = pkgs: with pkgs;
        callPackage
          (
            { ramalamaOverrides ? { }
            , llamaCppOverrides ? { }
            }:
              python3Packages.buildPythonPackage ({
                name = "ramalama";
                src = ./.;
                dependencies = [ (llama-cpp.override llamaCppOverrides) ];
              } // ramalamaOverrides)
          )
          { llamaCppOverrides.vulkanSupport = true; }
          ;

      ramalama = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          package = mkRamalama pkgs;
        in {
          inherit package;
          app = {
            type = "app";
            program = toString (pkgs.writeShellScript "ramalama" "${package}/bin/ramalama \"$@\"");
          };
        }
      );
    in {
      packages = forAllSystems (system: {
        ramalama = ramalama.${system}.package;
        default = ramalama.${system}.package;
      });

      apps = forAllSystems (system :{
        ramalama = ramalama.${system}.app;
        default = ramalama.${system}.app;
      });
    };
}