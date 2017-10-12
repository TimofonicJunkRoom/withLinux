// http://www.opengl-tutorial.org/beginners-tutorials/tutorial-1-opening-a-window/

#include <cstdio>
#include <cstdlib>

#include <GL/glew.h>
#include <GLFW/glfw3.h> // window and keyboard

int
main(void)
{
	// initialize GLFW
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW.\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // OpenGL 3.3
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		// don't use the old OpenGL

	// open a window and create its OpenGL context
	GLFWwindow* window = glfwCreateWindow(1024, 768,
			"Test", NULL, NULL);
	if (nullptr == window) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window); // initialize GLEW
	glewExperimental = true; // needed in core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Falied to init GLEW.\n");
		return -1;
	}

	// capture keys
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	do {
		glfwSwapBuffers(window);
		glfwPollEvents();
	} while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS
			&& glfwWindowShouldClose(window) == 0);

	return 0;
}
