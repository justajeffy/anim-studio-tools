#version 150 core

// VertexShader Uniforms
uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjMatrix;

// Input from VBOs
in vec3 iPosition;
in vec3 iNormal;
in vec2 iTexCoord;

// Output to FragmentShader
out vec3 oNormal;
out vec2 oTexCoord;

void main(void)
{
	mat4 mvpMatrix = ProjMatrix * ViewMatrix * ModelMatrix;
	gl_Position = mvpMatrix * vec4(iPosition, 1);
	//gl_Position /= gl_Position.w;
		
	mat3 normalMtx = mat3(ViewMatrix * ModelMatrix);
	oNormal = normalMtx * normalize(iNormal);
	//oNormal = normalize(iNormal.xyz);
	
	oTexCoord = iTexCoord;
	//oTexCoord = gl_Position.xy/gl_Position.w;	
}
