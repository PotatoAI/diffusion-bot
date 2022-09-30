{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
  nativeBuildInputs = [ pkgconfig ];
  buildInputs = [ openssl ffmpeg python39Packages.toml ];
  shellHook = ''
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
      pkgs.lib.makeLibraryPath [ stdenv.cc.cc.lib ]
    }"'';
}
