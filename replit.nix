{ pkgs }: {
  deps = [
    pkgs.libxcrypt
    pkgs.bash
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.python310Full
    pkgs.python310Packages.pip
    pkgs.python310Packages.uvicorn
    pkgs.python310Packages.fastapi
    pkgs.python310Packages.textblob
    pkgs.python310Packages.google-auth
    pkgs.python310Packages.google-api-python-client
    pkgs.python310Packages.textblob
  ];
}
