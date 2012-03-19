#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0;
uniform sampler2DShadow TextureSampler1; // shadow map

uniform vec3 LightDir;
uniform float ShadowBias;
uniform float ShadowBlurScale;

// Input from VertexShader
highp in vec3 oNormal;
highp in vec2 oTexCoord;
highp in vec4 oShdProjCoord;

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

	//outColor = texture2D(TextureSampler0, saturate( 0.5 + 0.5*oShdProjCoord.xy/oShdProjCoord.w )).xxxx;

	vec4 shdPJ = oShdProjCoord;
	shdPJ.xyz = saturate( 0.5 + 0.5 * shdPJ.xyz / shdPJ.w ); // todo via matrix multiply in vertex shader
	//shdPJ.z -= ShadowBias;
	shdPJ.w = 1;

	vec4 shadowK = textureProj( TextureSampler1, shdPJ ).rrrr; 
	//outColor *= shadowK;

	// blur test: apply a 25 taps poisson filter blur kernel
	vec4 shadowBlurScale = vec4( ShadowBlurScale, ShadowBlurScale, 1, 1 );
	
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.978698, -0.0884121, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.841121, 0.521165, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.71746, -0.50322, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.702933, 0.903134, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.663198, 0.15482, 0, 0) ).rrrr;
	
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.495102, -0.232887, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.364238, -0.961791, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.345866, -0.564379, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.325663, 0.64037, 0, 0) ).rrrr;
	
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.182714, 0.321329, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.142613, -0.0227363, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.0564287, -0.36729, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(-0.0185858, 0.918882, 0, 0) ).rrrr;
	
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.0381787, -0.728996, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.16599, 0.093112, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.253639, 0.719535, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.369549, -0.655019, 0, 0) ).rrrr;
	
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.423627, 0.429975, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.530747, -0.364971, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.566027, -0.940489, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.639332, 0.0284127, 0, 0) ).rrrr;
	
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.652089, 0.669668, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.773797, 0.345012, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.968871, 0.840449, 0, 0) ).rrrr;
	shadowK += textureProj( TextureSampler1, shdPJ + shadowBlurScale * vec4(0.991882, -0.657338, 0, 0) ).rrrr;

	shadowK /= 26;
	outColor *= shadowK;

	//outColor = oShdProjCoord;
}
