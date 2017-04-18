#!/bin/sh
# set -e
# show how to run a java program on linux

echo "removing all class files"
rm ./*.class
echo 

echo "compile the hello word program hello.java"
javac hello.java
echo 

echo "run the compiled program with java"
java Main

echo "making jar package"
jar cvfm hello.jar MANIFEST.MF -C . .
java -jar hello.jar
# See maven for better project management

# reference: http://docs.oracle.com/javase/tutorial/getStarted/cupojava/unix.html
