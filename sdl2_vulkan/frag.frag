#version 450

layout(location = 0) in vec3 frag_color;

layout(location = 0) out vec4 color;

void main() {
	color = vec4(frag_color, 1.0f);
}

