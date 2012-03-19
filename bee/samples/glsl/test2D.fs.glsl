#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0;

// Input from VertexShader
highp in vec2 oTexCoord;
highp in vec4 oColor;

// Final output color
highp out vec4 outColor;

void main(void)
{
	vec4 tex0 = texture2D(TextureSampler0, oTexCoord);

	outColor = oColor * tex0;
}
