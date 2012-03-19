#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0; // position
uniform sampler2D TextureSampler1; // normal
uniform sampler2D TextureSampler2; // baseColour
uniform sampler2D TextureSampler3; // occlusion

uniform int LightType;
uniform vec3 LightPosition;
uniform vec3 LightDirection;
uniform vec3 LightDiffuseColor;
uniform vec3 LightSpecularColor;
uniform float LightAttenuation;
uniform float LightConeAngle;
uniform float LightConeDelta;
uniform float LightConeRolloff;

uniform vec3 CameraPosition;

// Input from VertexShader
highp in vec2 oTexCoord;
highp in vec4 oColor;

// Final output color
highp out vec4 outColor;

void DefaultLightingModel( in vec3 i_Position
						 , in vec3 i_Normal
						 , in vec3 i_LightDirection
						 , in vec3 i_CameraPosition
						 , in float i_materialSpecularPower
						 , in float i_materialSpecularRoughness
						 , out vec3 o_DiffuseK
						 , out vec3 o_SpecularK
		)
{
	// Diffuse lighting model
	float NdotL = max( 0, dot( i_Normal, i_LightDirection ) );
	o_DiffuseK = vec3( NdotL );

	// Specular lighting model (Blinn)
	vec3 viewVector = normalize( i_CameraPosition - i_Position );
	vec3 halfVector = normalize( i_LightDirection + viewVector );
	float NdotH = max( 0, dot( i_Normal, halfVector ) );
	//float sNdotH = NdotH * max( 0, sign( NdotL ) ); // produce artefacts...

	o_SpecularK = vec3( pow( NdotH, i_materialSpecularPower ) / i_materialSpecularRoughness );
	//o_SpecularK = vec3( pow( sNdotH, i_materialSpecularPower ) / i_materialSpecularRoughness );
	//o_SpecularK = vec3( pow( NdotL * NdotH, i_materialSpecularPower ) / i_materialSpecularRoughness );
}

void main(void)
{
	vec2 invTexCoord = vec2( oTexCoord.x, 1 - oTexCoord.y );

	vec3 position = texture2D( TextureSampler0, invTexCoord ).rgb;
	vec3 normal = normalize( texture2D(TextureSampler1, invTexCoord ).rgb );
	vec3 baseColour = texture2D( TextureSampler2, invTexCoord ).rgb;
	vec3 occlusion = texture2D( TextureSampler3, invTexCoord ).rgb;

	vec3 diffuseColor;

	// todo: sample screen space textures
	vec3 specularColor = vec3( 0, 0, 0 );
	float materialSpecularPower = 5;
	float materialSpecularRoughness = 3;

	// todo: one shader for each light type !

	if ( LightType == 0 ) // Ambient
	{
		diffuseColor = LightDiffuseColor;
	}
	else if ( LightType == 1 ) // Point
	{
		vec3 lightDir = LightPosition - position;
		lightDir = normalize( lightDir );

		DefaultLightingModel( position, normal, lightDir, CameraPosition, materialSpecularPower, materialSpecularRoughness,
							  diffuseColor, specularColor );
	}
	else if ( LightType == 2 ) // Spot
	{
		vec3 lightDir = LightPosition - position;
		lightDir = normalize( lightDir );

		DefaultLightingModel( position, normal, lightDir, CameraPosition, materialSpecularPower, materialSpecularRoughness,
							  diffuseColor, specularColor );

		//float LdotL = dot( normalize(LightPosition - position), LightDirection );
		float LdotL = dot( lightDir, LightDirection );

		//float coneAtt = 1-saturate( ( (1-dot( lightDir, LightDirection )) - LightConeAngle ) / ( LightConeDelta - LightConeAngle ) );
		float coneAtt = smoothstep( LightConeDelta, LightConeAngle, saturate( LdotL ) );

		//float coneAtt = saturate( ( LdotL - LightConeAngle ) / ( LightConeDelta - LightConeAngle ) );

		diffuseColor *= coneAtt;

		//diffuseColor = LdotL.xxx;

		//diffuseColor = dot( lightDir, LightDirection ).xxx;

		//if ( NdotL < LightConeAngle )
		//	diffuseColor *= 0;
	}
	else if ( LightType == 3 ) // Area
	{
		diffuseColor = vec3(0,1,0);
	}
	else if ( LightType == 4 ) // Distant
	{
		DefaultLightingModel( position, normal, LightDirection, CameraPosition, materialSpecularPower, materialSpecularRoughness,
							  diffuseColor, specularColor );
	}

	//diffuseColor *= LightDiffuseColor;
	specularColor *= LightSpecularColor;

	outColor.rgb = diffuseColor * baseColour;
	outColor.rgb += specularColor;
	outColor.a = 1;
}
