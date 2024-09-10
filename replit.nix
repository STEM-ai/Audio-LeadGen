{ pkgs }: {
  deps = [
    pkgs.poppler_utils
    pkgs.rustc
    pkgs.cargo
    pkgs.libxcrypt
    pkgs.bash
    pkgs.libiconv
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.uvicorn
    pkgs.python310Packages.fastapi
    pkgs.python310Packages.google-auth
    pkgs.python310Packages.google-api-python-client
    pkgs.python310Packages.requests
    pkgs.python310Packages.textblob
  ];

  shellHook = ''
    pip install -r requirements.txt --constraint constraints.txt
  '';
}