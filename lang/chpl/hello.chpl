// http://chapel.cray.com/docs/latest/examples/
module Hello {
	config const message = "Hello, Chapel!";
	config const numMessages = 10;
	
	proc main() {
		// hello world
		writeln("Chapel!");

		// data parallel
		forall msg in 1..numMessages do
			writeln(message, " from ", msg, " of ", numMessages);
	}
}
