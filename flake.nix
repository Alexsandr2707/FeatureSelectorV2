{
  description = "Dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
    python = pkgs.python313.withPackages (ps: with ps; [
        numpy 
        pandas
        matplotlib
        torch
        scikit-learn
        pydantic

        rich
        openpyxl
        ipykernel # only for notebook support
    ]);
  in
  {
    devShells."${system}".default = pkgs.mkShell {
      buildInputs = with pkgs; [
        # Python
        python

        # utils
        git
        hello
        tree
      ];

      shellHook = ''
        echo "Dev environment loaded"
      '';
    };
  };
}
