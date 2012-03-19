#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0; // viewPos
uniform sampler2D TextureSampler1; // viewNormal
uniform sampler2D TextureSampler2; // albedo

uniform vec4 Light0PositionAttenuation;
uniform vec3 Light0Color;
uniform mat4 ViewMatrix;

// Input from VertexShader
highp in vec2 oTexCoord;
highp in vec4 oColor;

// Final output color
highp out vec4 outColor;

void main(void)
{
	vec3 viewPosition = texture2D(TextureSampler0, oTexCoord).rgb;
	vec3 viewNormal = texture2D(TextureSampler1, oTexCoord).rgb;
	vec3 albedo = texture2D(TextureSampler2, oTexCoord).rgb;
	vec3 norViewNormal = normalize(viewNormal);
	
	vec3 emissive = vec3(0,0,0);
	
	vec3 LightDir = Light0PositionAttenuation.xyz - viewPosition;
	float lightDist = length( LightDir );
	LightDir = normalize( LightDir );
	float lightAtt = 1 - lightDist * Light0PositionAttenuation.w;
	
	outColor.rgb = emissive;
	outColor.rgb += vec3( max(0, dot(norViewNormal, LightDir)) ) * lightAtt * Light0Color;

	outColor.rgb *= albedo;

	// add some basic fog
	/*float viewDist = length(viewPosition);
	vec3 fogColor = vec3(1,1,1);
	float fogFarDist = 2000;
	float fogAtt = saturate( viewDist / fogFarDist );
	outColor.rgb = lerp(outColor.rgb, fogColor, fogAtt);*/

	// fake ao test
	//float lv = length(viewNormal);
	//float fakeAo = saturate( lv*lv );
	//outColor.rgb = vec3(fakeAo, fakeAo, fakeAo);

	//outColor.rgb = lightAtt.xxx;
	//outColor.rgb = fogAtt.xxx;
	//outColor.rgb = dot(norViewNormal, LightDir).xxx;
	//outColor.rgb = dot(norViewNormal, vec3(0,0,1)).xxx;
	//outColor.rgb = Light0PositionAttenuation.rgb;
	outColor.a = 1;
}
