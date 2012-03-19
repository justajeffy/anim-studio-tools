#version 150 core

// Input from VBOs
in vec3 iPosition;
in vec2 iTexCoord;

// Output to FragmentShader
out vec2 oTexCoord;

void main(void)
{
	gl_Position = vec4(iPosition, 1);
		
	oTexCoord = iTexCoord;
}
