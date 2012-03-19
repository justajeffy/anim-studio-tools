#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0; // viewPos
uniform sampler2D TextureSampler1; // viewNormal
uniform vec4 TextureSamplerSize1;
uniform sampler2D TextureSampler2; // albedo

// Tonal Art Maps
uniform sampler2D TextureSampler3;
uniform sampler2D TextureSampler4;
uniform sampler2D TextureSampler5;
uniform sampler2D TextureSampler6;
uniform sampler2D TextureSampler7;
uniform sampler2D TextureSampler8;

uniform vec4 Light0PositionAttenuation;
uniform vec3 Light0Color;
uniform mat4 ViewMatrix;

// Input from VertexShader
highp in vec2 oTexCoord;
highp in vec4 oColor;

// Final output color
highp out vec4 outColor;

float Thickness = 1.5f;
float brightThreshold = 0.86;
float averageThreshold = 0.53;
vec3 darkColor = vec3(0.2,0.2,0.2);
vec3 averageColor = vec3(0.5,0.5,0.5);
vec3 brightColor = vec3(0.8,0.8,0.8);

vec3 ConvertToToonLightingModel( float a_Lighting )
{
	return a_Lighting.xxx; // comment to use "toon lighting"
	if ( a_Lighting > brightThreshold ) return brightColor;
	if ( a_Lighting > averageThreshold ) return averageColor;
	return darkColor;
}

void main(void)
{
	vec4 viewPositionU = texture2D(TextureSampler0, oTexCoord);
	vec4 viewNormalV = texture2D(TextureSampler1, oTexCoord);
	vec3 albedo = texture2D(TextureSampler2, oTexCoord).rgb;

	vec3 viewPosition = viewPositionU.xyz;
	vec3 viewNormal = viewNormalV.xyz;
	vec2 orgUV = vec2( viewPositionU.w, viewNormalV.w );

	viewNormal = normalize(viewNormal);
	vec3 lightDir = vec3(1,0,1);
	float emissive = 0.3;
	float lighting = saturate(emissive + dot(viewNormal, lightDir));

	outColor.rgb = albedo * ConvertToToonLightingModel( lighting );
	//outColor.rgb = ConvertToToonLightingModel( lighting );

	vec2 rtSize = TextureSamplerSize1.zw;

	vec3 vpNormal1 = normalize( texture2D(TextureSampler1, saturate(oTexCoord + rtSize * vec2( -Thickness, -Thickness )) ).rgb );
	vec3 vpNormal2 = normalize( texture2D(TextureSampler1, saturate(oTexCoord + rtSize * vec2( -Thickness, +Thickness )) ).rgb );
	vec3 vpNormal3 = normalize( texture2D(TextureSampler1, saturate(oTexCoord + rtSize * vec2( +Thickness, -Thickness )) ).rgb );
	vec3 vpNormal4 = normalize( texture2D(TextureSampler1, saturate(oTexCoord + rtSize * vec2( +Thickness, +Thickness )) ).rgb );

	float dp1 = abs( dot( viewNormal, vpNormal1 ) );
	float dp2 = abs( dot( viewNormal, vpNormal2 ) );
	float dp3 = abs( dot( viewNormal, vpNormal3 ) );
	float dp4 = abs( dot( viewNormal, vpNormal4 ) );

	float avDp = (dp1 + dp2 + dp3 + dp4) * 0.25;
	avDp = avDp*avDp;

	outColor.rgb *= (avDp*avDp).xxx;

	//outColor.rgb = (avDp*avDp).xxx;
	//outColor.rgb = texture2D( TextureSampler3, orgUV * 10 ).rgb;

	// add TAM hatching
	float tamLighting = lighting * 6;
	vec2 tamScale = vec2( 5, 5 ); // use other coordinates?
	//vec2 tamUv2Use = oTexCoord;
	vec2 tamUv2Use = orgUV;
	vec3 tam0 = texture2D( TextureSampler3, tamUv2Use * tamScale ).rgb;
	vec3 tam1 = texture2D( TextureSampler4, tamUv2Use * tamScale ).rgb;
	vec3 tam2 = texture2D( TextureSampler5, tamUv2Use * tamScale ).rgb;
	vec3 tam3 = texture2D( TextureSampler6, tamUv2Use * tamScale ).rgb;
	vec3 tam4 = texture2D( TextureSampler7, tamUv2Use * tamScale ).rgb;
	vec3 tam5 = texture2D( TextureSampler8, tamUv2Use * tamScale ).rgb;

	vec3 tamC = lerp( tam5, tam4, saturate( tamLighting ) );
	tamC = lerp( tamC, tam3, saturate( tamLighting - 1 ) );
	tamC = lerp( tamC, tam2, saturate( tamLighting - 2 ) );
	tamC = lerp( tamC, tam1, saturate( tamLighting - 3 ) );
	tamC = lerp( tamC, tam0, saturate( tamLighting - 4 ) );
	tamC = lerp( tamC, vec3(1,1,1), saturate( tamLighting - 5 ) );

	outColor.rgb *= tamC;
}
