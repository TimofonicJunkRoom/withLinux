Rebuilding this dwm package
===

`dwm 6.1-3`

* copy patches to `debian/patches`: `alpha.patch`

* remove `debian/local/*`

* copy my config to `debian/local`: `cp ... local/config.my.h`

* edit `debian/rules`

```patch
diff --git a/debian/rules b/debian/rules
index ae9cd41..4b4e994 100755
--- a/debian/rules
+++ b/debian/rules
@@ -21,6 +21,7 @@ override_dh_auto_install:
        for ALTERNATIVE in debian/local/config.*.h; \
        do \
                $(MAKE) clean; \
+               patch -p1 < debian/patches/alpha.patch; \
                cp $${ALTERNATIVE} config.h; \
                CFLAGS="$(CFLAGS)" $(MAKE) PREFIX=/usr; \
                install -m 0755 dwm debian/dwm/usr/bin/dwm.`basename $${ALTERNATIVE} | cut -d'.' -f 2`; \
```

* build package `debuild`, and install `sudo debi`

* switch dwm alternative `sudo update-alternatives --config dwm`.

ref: https://wiki.archlinux.org/index.php/Dwm

ref: https://wiki.archlinux.org/index.php/Desktop_entries
