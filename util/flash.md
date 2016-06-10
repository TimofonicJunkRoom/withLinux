Flash player for browser on Debian
---

> https://wiki.debian.org/PepperFlashPlayer/Installing  

# package

```
apt install pepperflashplugin-nonfree browser-plugin-freshplayer-pepperflash
update-pepperflashplugin-nonfree --install
update-pepperflashplugin-nonfree --status
```

# manual setup for chrome

according to script `update-pepperflashplugin-nonfree`,
first download the newest deb file of chrome
```
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
```
then unpack it
```
241             dpkg-deb -x $debfile unpackchrome
```
next, find the shared object
```
243             sofile=unpackchrome/opt/google/chrome/PepperFlash/libpepflashplayer.so
244             [ -e $sofile ] || sofile=unpackchrome/opt/google/chrome-unstable/PepperFlash/libpepflashplayer.so
245             [ -e $sofile ] || sofile=unpackchrome/opt/google/chrome-beta/PepperFlash/libpepflashplayer.so
246             [ -e $sofile ] || sofile=unpackchrome/opt/google/chrome/PepperFlash/libpepflashplayer.so
```
and install
```
248             mv -f $sofile /usr/lib/pepperflashplugin-nonfree
249             chown root:root /usr/lib/pepperflashplugin-nonfree/libpepflashplayer.so
250             chmod 644 /usr/lib/pepperflashplugin-nonfree/libpepflashplayer.so
```

But that's not all, another file is needed
```
252             jsonfile=unpackchrome/opt/google/chrome/PepperFlash/manifest.json
253             [ -e $jsonfile ] || jsonfile=unpackchrome/opt/google/chrome-unstable/PepperFlash/manifest.json
254             [ -e $jsonfile ] || jsonfile=unpackchrome/opt/google/chrome-beta/PepperFlash/manifest.json
255             [ -e $jsonfile ] || jsonfile=unpackchrome/opt/google/chrome/PepperFlash/manifest.json
256 
257             if [ -e $jsonfile ]
258             then
259                 mv -f $jsonfile /usr/lib/pepperflashplugin-nonfree
260                 chown root:root /usr/lib/pepperflashplugin-nonfree/manifest.json
261                 chmod 644 /usr/lib/pepperflashplugin-nonfree/manifest.json
262             fi
```
