#!/usr/bin/env bash

# get the current version

if [[ $(git diff --stat) != '' ]]; then
	echo "[!] Working tree is dirty; aborting."
	exit 1
fi

if [ $# -ne 1 ]; then
	echo "[!] Please provide new version"
	exit 1
fi

currentversion=$(python3 -c "import nmfu; print(nmfu.__version__)")
newversion=$1
root=$(git rev-parse --show-toplevel)

echo "[*] Current version is ${currentversion}"
echo "[*] New version is ${newversion}"

sed -i "s/__version__ = \"${currentversion}\"/__version__ = \"${newversion}\"/" ${root}/nmfu.py
sed -i "s/\(version: \|nmfu-\)${currentversion}/\1${newversion}/" ${root}/AppImageBuilder.yml

if [ $DRY_RUN ]; then
	echo "[!] Not creating commits because dry run"
else
	echo "[*] Creating commit"
	git add ${root}/nmfu.py ${root}/AppImageBuilder.yml
	git commit -m "bump version to ${newversion}"
	echo "[*] Creating tag"
	git tag -a -m "version ${newversion}" v${newversion}
fi
