#version 330

in vec2 texCoord0;
in vec3 normal0;
in vec3 color0;
in vec3 position0;

uniform vec4 lightColor;
uniform sampler2D sampler1;
uniform sampler2D sampler2;
uniform sampler2D sampler3;
uniform sampler2D sampler4;
uniform sampler2D sampler5;
uniform vec4 lightDirection;

out vec4 Color;

void main()
{
	Color =  clamp(texture(sampler1, texCoord0)*0.8 + texture(sampler2, texCoord0)*0.5 +texture(sampler3, texCoord0)*0.3 + texture(sampler4, texCoord0)*0.2 + texture(sampler5, texCoord0)*0.1,0,1); //you must have Color
}
