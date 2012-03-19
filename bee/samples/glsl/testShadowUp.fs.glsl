#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0;
uniform sampler2D TextureSampler1;
uniform sampler2D TextureSampler2;
uniform sampler2D TextureSampler3;
uniform sampler2D TextureSampler4;
uniform sampler2D TextureSampler5;
uniform sampler2D TextureSampler6;
uniform sampler2D TextureSampler7;

uniform vec3 LightDir;

// Input from VertexShader
highp in vec3 oNormal;
highp in vec2 oTexCoord;

// Final output color
highp out vec4 outColor;

void main(void)
{
	//outColor = vec4(oNormal, 1.0)*0.5+0.5;

	vec4 emissive = vec4(0.3,0.3,0.3, 1);

	outColor = emissive;
	outColor += vec4(dot(normalize( oNormal ), LightDir));

	vec4 tex0 = texture2D(TextureSampler0, oTexCoord);
	outColor *= tex0;

	outColor = vec4(1,1,1,1);
}
