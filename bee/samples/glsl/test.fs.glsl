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

	float scale = 8;
	vec4 tex0 = texture2D(TextureSampler0, oTexCoord);
	vec4 tex1 = texture2D(TextureSampler1, oTexCoord * vec2(2, 2) );
	vec4 tex2 = texture2D(TextureSampler2, oTexCoord * vec2(4, 4) );
	vec4 tex3 = texture2D(TextureSampler3, oTexCoord * vec2(8, 8) );
	vec4 tex4 = texture2D(TextureSampler4, oTexCoord * vec2(16, 16) );
	vec4 tex5 = texture2D(TextureSampler5, oTexCoord * vec2(32, 32) );
	vec4 tex6 = texture2D(TextureSampler6, oTexCoord * vec2(64, 64) );
	vec4 tex7 = texture2D(TextureSampler7, oTexCoord * vec2(128, 128) );

	outColor *= (tex0 + tex1 + tex2 + tex3 + tex4 + tex5 + tex6 + tex7) / scale;
}
