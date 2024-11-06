{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.google-cloud-sdk
    pkgs.python310
    pkgs.python310Packages.requests
    pkgs.python310Packages.uvicorn
    pkgs.python310Packages.cartesia
    (pkgs.python310Packages.buildPythonPackage rec {
      pname = "textblob";
      version = "0.15.3";

      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "0f1b3da8d869654a45a34674e4d29b0d5e8bddf4ea0e5ff7cfea5db6f3c1fc14";
      };

      propagatedBuildInputs = with pkgs.python310Packages; [ nltk numpy six ];

      meta = with pkgs.lib; {
        description = "Simple, Pythonic text processing.";
        homepage = "https://textblob.readthedocs.io";
        license = licenses.mit;
      };
    })
  ];

  shellHook = ''
    echo "Shell environment set up."
  '';
}