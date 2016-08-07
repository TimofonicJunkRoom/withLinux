main:
	$(CC) -shared -fPIC -g -Wall kernel.c -o kernel.so
	python3 myffi.py
clean:
	$(RM) kernel.so
