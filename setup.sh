sudo dnf install python3.10 python3-pip gcc pipenv -y
su -curl https://pyenv.run | bash
export PIPENV_YES=1
pipenv install
