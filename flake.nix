{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          uv
          openssl
          cudatoolkit
          nvidia-docker
        ];
        buildInputs = with pkgs; [
          openssl
        ];
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib
          openssl
          cudatoolkit
          nvidia-docker
          libGL
          glib
          glibc.bin
        ]);
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        shellHook = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH
        '';
      };
    };
}

