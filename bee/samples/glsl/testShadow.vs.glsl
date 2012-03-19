#version 150 core

// VertexShader Uniforms
uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjMatrix;
uniform mat4 ShadowViewMatrix;
uniform mat4 ShadowProjMatrix;

// Input from VBOs
in vec3 iPosition;
in vec3 iNormal;
in vec2 iTexCoord;

// Output to FragmentShader
out vec3 oNormal;
out vec2 oTexCoord;
out vec4 oShdProjCoord; 

void main(void)
{
	mat4 mvpMatrix = ProjMatrix * ViewMatrix * ModelMatrix;
	gl_Position = mvpMatrix * vec4(iPosition, 1);
		
	mat3 normalMtx = mat3(ViewMatrix * ModelMatrix);
	oNormal = normalMtx * normalize(iNormal);
	oTexCoord = iTexCoord;

	mat4 shmvpMatrix = ShadowProjMatrix * ShadowViewMatrix * ModelMatrix;
	oShdProjCoord = shmvpMatrix * vec4(iPosition, 1);

}
