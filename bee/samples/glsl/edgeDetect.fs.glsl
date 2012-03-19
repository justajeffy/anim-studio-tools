#version 150 core

// FragmentShader Uniforms
uniform sampler2D TextureSampler0;
uniform vec4 TextureSamplerSize0;

// Input from VertexShader
highp in vec2 oTexCoord;
highp in vec4 oColor;

// Final output color
highp out vec4 outColor;

float getGray(vec4 c)
{
    return(dot(c.rgb,((0.33333).xxx)));
}

// Use these two variables to set the outline properties( thickness and threshold )
float Thickness = 2.5f;
float Threshold = 0.2f;

void main(void)
{
	vec2 rtSize = TextureSamplerSize0.zw;

	vec2 Tex = oTexCoord;
	vec4 Color = texture2D(TextureSampler0, Tex);

	vec2 ox = vec2(Thickness * rtSize.x, 0);
	vec2 oy = vec2(0, Thickness * rtSize.y);
	vec2 uv = Tex.xy;
	vec2 PP = uv - oy;

	vec4 CC = texture2D(TextureSampler0,PP - ox); float g00 = getGray(CC);

	CC = texture2D(TextureSampler0,PP); float g01 = getGray(CC);
	CC = texture2D(TextureSampler0,PP + ox); float g02 = getGray(CC);
	PP = uv;

	CC = texture2D(TextureSampler0,PP - ox); float g10 = getGray(CC);
	CC = texture2D(TextureSampler0,PP); float g11 = getGray(CC);
	CC = texture2D(TextureSampler0,PP + ox); float g12 = getGray(CC);
	PP = uv + oy;

	CC = texture2D(TextureSampler0,PP - ox); float g20 = getGray(CC);
	CC = texture2D(TextureSampler0,PP); float g21 = getGray(CC);
	CC = texture2D(TextureSampler0,PP + ox); float g22 = getGray(CC);

	float K00 = -1;
	float K01 = -2;
	float K02 = -1;
	float K10 = 0;
	float K11 = 0;
	float K12 = 0;
	float K20 = 1;
	float K21 = 2;
	float K22 = 1;
	float sx = 0;
	float sy = 0;
	sx += g00 * K00;
	sx += g01 * K01;
	sx += g02 * K02;
	sx += g10 * K10;
	sx += g11 * K11;
	sx += g12 * K12;
	sx += g20 * K20;
	sx += g21 * K21;
	sx += g22 * K22;
	sy += g00 * K00;
	sy += g01 * K10;
	sy += g02 * K20;
	sy += g10 * K01;
	sy += g11 * K11;
	sy += g12 * K21;
	sy += g20 * K02;
	sy += g21 * K12;
	sy += g22 * K22;
	float dist = sqrt(sx * sx + sy * sy);
	float result = 1;
	if (dist > Threshold) result = 0;

	// The scene will be in black and white, so to render
	// everything normaly, except for the edges, bultiply the
	// edge texture with the scenecolor
	//outColor = Color * result.xxxx;

	outColor = result.xxxx;

}
