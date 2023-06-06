//=============================================================================================
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Demeter Abel Bence
// Neptun : HHUZ6K
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(_fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(_fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 position;
	vec3 Le;
	Light(vec3 _position, vec3 _Le) {
		position = _position;
		Le = _Le;
	}
	Light() { }
};

struct Triangle : public Intersectable {
	vec3 a, b, c;

	Triangle(const vec3& _a, const vec3& _b, const vec3& _c, Material* _material) {
		a = _a;
		b = _b;
		c = _c;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec3 vector1 = b - a;
		vec3 vector2 = c - a;
		vec3 normal = normalize(cross(vector1, vector2));

		float t = dot((a - ray.start), normal) / dot(ray.dir, normal);
		if (t < 0) return hit;

		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;

		vec3 cross1 = cross(b - a, hit.position - a);
		vec3 cross2 = cross(c - b, hit.position - b);
		vec3 cross3 = cross(a - c, hit.position - c);
		float dot1 = dot(cross1, normal);
		float dot2 = dot(cross2, normal);
		float dot3 = dot(cross3, normal);
		if (dot1 < 0 || dot2 < 0 || dot3 < 0)
			return Hit();

		vector1 = b - hit.position;
		vector2 = c - hit.position;

		hit.normal = normalize(cross(vector1, vector2));
		hit.material = material;

		return hit;
	}

	void rotateVector(vec3& r, float angle, const vec3& d) {
		vec4 tmp;
		tmp.x = r.x; tmp.y = r.y; tmp.z = r.z; tmp.w = 0;
		tmp = tmp * RotationMatrix(angle, d);
		r.x = tmp.x; r.y = tmp.y; r.z = tmp.z;
	}

	void rotate(float angle, const vec3& d) {
		rotateVector(a, angle, d);
		rotateVector(b, angle, d); 
		rotateVector(c, angle, d);
	}

	void moveX(float _x) {
		a.x += _x;
		b.x += _x;
		c.x += _x;
	}

	void moveY(float _y) {
		a.y += _y;
		b.y += _y;
		c.y += _y;
	}
};

struct Cone : public Intersectable{
	vec3 p;
	vec3 n;
	float angle;
	float h;

	Light* light;

	Cone() {}
	Cone(const vec3& _p, const vec3& _n, float _angle, float _h, Material* m) {
		p = _p;
		n = _n;
		angle = _angle;
		h = _h;
		material = m;

		light = new Light();
	}

	void setLight(const vec3& _position, const vec3& _Le) {
		light->position = _position;
		light->Le = _Le;
	}

	Hit intersect(const Ray& ray) {
		vec3 s = ray.start;
		vec3 d = ray.dir;
		vec3 k = s - p;
		float cos = cosf(angle) * cosf(angle);

		float a = dot(d, n) * dot(d, n) - dot(d, d) * cos;
		float b = 2 * dot(d, n) * dot(k, n) - cos * 2 * (d.x * k.x + d.y * k.y + d.z * k.z);
		float c = dot(k, n) * dot(k, n) - cos * dot(k, k);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return Hit();
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return Hit();

		Hit hit;
		vec3 r1 = ray.start + ray.dir * t1;
		vec3 r2 = ray.start + ray.dir * t2;

		if (dot((r1 - p), n) < 0 || dot((r1 - p), n) > h) {
			if (dot((r2 - p), n) < 0 || dot((r2 - p), n) > h)
				return Hit();	
			else {
				hit.material = material;
				hit.t = t2;
				hit.position = r2;
				hit.normal = 2 * (dot(r2 - p, n)) * n - 2 * (r2 - p) * cos;
				return hit;
			}
		}
		else {
			hit.material = material;
			hit.t = t1;
			hit.position = r1;
			hit.normal = 2 * (dot(r1 - p, n)) * n - 2 * (r1 - p) * cos;
			return hit;
		}
	}

	void rotateVector(vec3& r, float angle, const vec3& d) {
		vec4 tmp;
		tmp.x = r.x; tmp.y = r.y; tmp.z = r.z; tmp.w = 0;
		tmp = tmp * RotationMatrix(angle, d);
		r.x = tmp.x; r.y = tmp.y; r.z = tmp.z;
	}

	void rotate(float angle, const vec3& d) {
		rotateVector(p, angle, d);
		rotateVector(n, angle, d);
		rotateVector(light->position, angle, d);
	}
	void moveY(float _y) {
		p = p + _y;
		n = n + _y;
	}
	void moveX(float _x) {
		p = p + _x;
		p = p + _x;
	}

	~Cone() {
		delete light;
	}
};

struct Object {
	std::vector<Triangle*> triangles;
	Material* material;

	Object() {}
	Object(Material* m) { material = m; }
	int getNumberOfTriangles() const { return triangles.size(); }
	void addTriangle(const vec3& a, const vec3& b, const vec3& c) {
		Triangle* t = new Triangle(a, b, c, material);
		triangles.push_back(t);
	}
	void rotate(float angle, const vec3& d) {
		int numberOfTriangles = triangles.size();
		for (int i = 0; i < numberOfTriangles; ++i)
			triangles[i]->rotate(angle, d);
	}
	void moveY(float _y) {
		int numberOfTriangles = triangles.size();
		for (int i = 0; i < numberOfTriangles; ++i)
			triangles[i]->moveY(_y);
	}
	void moveX(float _x) {
		int numberOfTriangles = triangles.size();
		for (int i = 0; i < numberOfTriangles; ++i)
			triangles[i]->moveX(_x);
	}
	~Object() {
		int numberOfTriangles = triangles.size();
		for (int i = 0; i < numberOfTriangles; ++i)
			delete triangles[i];
	}
};

struct Room {
	std::vector<Triangle*> visibleTriangles;
	std::vector<Cone*> cones;

	Object* walls;
	Object* diamond;
	Object* tetrahedron;

	Material* material;

	const float roomSize = 2.0f;
	const float coneLength = 0.8f;
	const float coneAngle = M_PI / 12;

	Room() {}
	void setMaterial(Material* m) { material = m; }
	void initialize() {

		walls = new Object(material);
		walls->addTriangle(vec3(-roomSize, -roomSize, -roomSize), vec3(-roomSize, -roomSize, roomSize), vec3(roomSize, -roomSize, -roomSize));
		walls->addTriangle(vec3(roomSize, -roomSize, -roomSize), vec3(roomSize, -roomSize, roomSize), vec3(-roomSize, -roomSize, roomSize));

		walls->addTriangle(vec3(-roomSize, roomSize, -roomSize), vec3(-roomSize, roomSize, roomSize), vec3(roomSize, roomSize, roomSize));
		walls->addTriangle(vec3(-roomSize, roomSize, -roomSize), vec3(roomSize, roomSize, roomSize), vec3(roomSize, roomSize, -roomSize));

		walls->addTriangle(vec3(-roomSize, -roomSize, -roomSize), vec3(-roomSize, -roomSize, roomSize), vec3(-roomSize, roomSize, -roomSize));
		walls->addTriangle(vec3(-roomSize, roomSize, -roomSize), vec3(-roomSize, -roomSize, roomSize), vec3(-roomSize, roomSize, roomSize));
		walls->addTriangle(vec3(-roomSize, -roomSize, roomSize), vec3(roomSize, -roomSize, roomSize), vec3(-roomSize, roomSize, roomSize));
		walls->addTriangle(vec3(-roomSize, roomSize, roomSize), vec3(roomSize, -roomSize, roomSize), vec3(roomSize, roomSize, roomSize));
		walls->addTriangle(vec3(roomSize, roomSize, -roomSize), vec3(roomSize, roomSize, roomSize), vec3(roomSize, -roomSize, roomSize));
		walls->addTriangle(vec3(roomSize, roomSize, -roomSize), vec3(roomSize, -roomSize, roomSize), vec3(roomSize, -roomSize, -roomSize));
		
		walls->addTriangle(vec3(-roomSize, -roomSize, -roomSize), vec3(-roomSize, roomSize, -roomSize), vec3(roomSize, roomSize, -roomSize));
		walls->addTriangle(vec3(-roomSize, -roomSize, -roomSize), vec3(roomSize, roomSize, -roomSize), vec3(roomSize, -roomSize, -roomSize));
		walls->rotate(M_PI / 6, vec3(0, 1, 0));

		diamond = new Object(material);
		diamond->addTriangle(vec3(0, 0, 0.78), vec3(0.45, 0.45, 0), vec3(0.45, -0.45, 0));
		diamond->addTriangle(vec3(0, 0, 0.78), vec3(0.45, 0. - 0.45, 0), vec3(-0.45, -0.45, 0));
		diamond->addTriangle(vec3(0, 0, 0.78), vec3(-0.45, -0.45, 0), vec3(0, 0, -0.78));
		diamond->addTriangle(vec3(0, 0, 0.78), vec3(-0.45, 0.45, 0), vec3(0.45, 0.45, 0));
		diamond->addTriangle(vec3(0, 0, -0.78), vec3(-0.45, 0.45, 0), vec3(-0.45, -0.45, 0));
		diamond->addTriangle(vec3(0, 0, -0.78), vec3(-0.45, 0. - 0.45, 0), vec3(0.45, -0.45, 0));
		diamond->addTriangle(vec3(0, 0, -0.78), vec3(0.45, 0. - 0.45, 0), vec3(0.45, 0.45, 0));
		diamond->addTriangle(vec3(0, 0, -0.78), vec3(0.45, 0.45, 0), vec3(0, 0, 0.78));
		diamond->addTriangle(vec3(0, 0, -0.78), vec3(0, 0, 0.78), vec3(-0.45, 0.45, 0));
		diamond->rotate(M_PI / 2, vec3(1, 0, 0));
		diamond->moveX(-1.0f);
		diamond->moveY(-1.0f);

		tetrahedron = new Object(material);
		tetrahedron->addTriangle(vec3(0, 0, 0), vec3(0, 1.5f, 0), vec3(1.5f, 0, 0));
		tetrahedron->addTriangle(vec3(0, 0, 0), vec3(0, 0, 1.5f), vec3(0, 1.5f, 0));
		tetrahedron->addTriangle(vec3(0, 0, 0), vec3(1.5f, 0, 0), vec3(0, 0, 1.5f));
		tetrahedron->addTriangle(vec3(1.5f, 0, 0), vec3(0, 1.5f, 0), vec3(0, 0, 1.5f));
		tetrahedron->rotate(M_PI / 3, vec3(0, 1, 0));
		tetrahedron->moveY(-2.0f);

		vec3 p, n, _n;
		vec3 lightPosition;
		
		//red cone
		p = vec3(roomSize, roomSize, roomSize);
		_n = vec3(0, 0, 0.0f);
		n = _n - p;
		n = normalize(n);
		Cone* cone1 = new Cone(p, n, coneAngle, coneLength, material);
		lightPosition = p + 0.5f * n;
		vec3 lightColor1 = vec3(1.0f, 0.0f, 0.0f);
		cone1->setLight(lightPosition, lightColor1);
		cone1->rotate(M_PI / 6, vec3(0, 1, 0));
		cones.push_back(cone1);

		//green cone
		p = vec3(roomSize, -roomSize, roomSize);
		_n = vec3(0, 0, 0.0f);
		n = _n - p;
		n = normalize(n);
		Cone* cone2 = new Cone(p, n, coneAngle, coneLength, material);
		lightPosition = p + 0.5f * n;
		vec3 lightColor2 = vec3(0.0f, 1.0f, 0.0f);
		cone2->setLight(lightPosition, lightColor2);
		cone2->rotate(M_PI / 6, vec3(0, 1, 0));
		cones.push_back(cone2);

		//blue cone
		p = vec3(-roomSize + 0.3f, -roomSize + 0.3f , roomSize);
		_n = vec3(0, -roomSize + 0.3f, 0);
		n = _n - p;
		n = normalize(n);
		Cone* cone3 = new Cone(p, n, coneAngle, coneLength, material);
		lightPosition = p + 0.5f * n;
		vec3 lightColor3 = vec3(0.0f, 0.0f, 1.0f);
		cone3->setLight(lightPosition, lightColor3);
		cone3->rotate(M_PI / 6, vec3(0, 1, 0));
		cones.push_back(cone3);
	}

	int visibleTriangleCount() const { return visibleTriangles.size(); }
	void updateVisibleWalls(const vec3& eye) {
		visibleTriangles.clear();

		int size = walls->triangles.size();
		for (int i = 0; i < size; ++i) {
			vec3 a = walls->triangles[i]->a;
			vec3 b = walls->triangles[i]->b;
			vec3 c = walls->triangles[i]->c;

			vec3 vector1 = b - a;
			vec3 vector2 = c - a;
			vec3 n = normalize(cross(vector1, vector2));

			float angle = acosf(dot(normalize(eye), n));

			if (abs(angle) >= M_PI / 2)
				visibleTriangles.push_back(walls->triangles[i]);

		}
	}
	void rotate(float angle, const vec3& d) {
		walls->rotate(angle, d);
		diamond->rotate(angle, d);
		tetrahedron->rotate(angle, d);
		
		int size = cones.size();
		for (int i = 0; i < size; ++i)
			cones[i]->rotate(angle, d);
	}
	void replaceClosestCone(const Hit& hit) {
		float min = 100;
		int size = cones.size();
		int index = -1;

		for (int i = 0; i < size; ++i) {
			vec3 v = hit.position - cones[i]->p;
			float l = length(v);

			if (l < min) {
				min = l;
				index = i;
			}
		}

		if (index > -1) {
			vec3 p = hit.position;
			vec3 n = hit.normal;
			cones[index]->p = p;
			cones[index]->n = normalize(n);
			cones[index]->light->position = p + 0.5f * n;
		}
	}

	~Room() {
		delete walls;
		delete diamond;
		delete tetrahedron;

		int size = cones.size();
		for (int i = 0; i < size; ++i)
			delete cones[i];
	}
};

const float epsilon = 0.01f;

class Scene {
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;

public:
	Room room;

	void update() { room.updateVisibleWalls(vec3(0, 0, -10)); }

	void setNewConePosition(int X, int Y) {	
		Hit hit = firstIntersect(camera.getRay(X, Y));
		if (hit.t < 0) return;
		room.replaceClosestCone(hit);
	}

	void build() {
		vec3 eye = vec3(0, 0, -10.0f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.0f, 0.0f, 0.0f);

		vec3 kd(0.0f, 0.0f, 0.0f), ks(0, 0, 0);
		Material* material = new Material(kd, ks, 50);

		room.setMaterial(material);
		room.initialize(); 

		int size = room.cones.size();
		for (int i = 0; i < size; ++i) {
			Light* l = room.cones[i]->light;
			lights.push_back(l);
		}
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;

		int size = room.cones.size();
		for (int i = 0; i < size; ++i) {
			Hit hit = room.cones[i]->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		
		size = room.visibleTriangleCount();
		for (int i = 0; i < size; ++i) {
			Hit hit = room.visibleTriangles[i]->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}

		size = room.diamond->getNumberOfTriangles();
		for (int i = 0; i < size; ++i) {
			Hit hit = room.diamond->triangles[i]->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}

		size = room.tetrahedron->getNumberOfTriangles();
		for (int i = 0; i < size; ++i) {
			Hit hit = room.tetrahedron->triangles[i]->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}

		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		vec3 N = normalize(hit.normal);
		vec3 V = -1 * ray.dir;
		float L = 0.2 * (1 + dot(N, V));

		vec3 outRadiance = vec3(L, L, L);

		for (int i = 0; i < room.cones.size(); ++i) {
			Light* light = room.cones[i]->light;
			vec3 Ll = light->position - hit.position;



			Ray shadowRay(hit.position + hit.normal * epsilon, Ll);
			Hit shadowHit = firstIntersect(shadowRay);

			float distance = length(hit.position - light->position);
			if (shadowHit.t < 0 || shadowHit.t > distance) {
				outRadiance = outRadiance + light->Le * (1 / (distance * distance));
			}
		}

		return outRadiance;
	}
	void animate(float dt) { camera.animate(dt); }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a')
		scene.room.rotate(M_PI / 20, vec3(0.0f, 1.0f, 0.0f));
	if (key == 'd')
		scene.room.rotate(-M_PI / 20, vec3(0.0f, 1.0f, 0.0f));

	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {

	bool press = false;

	switch (state) {
		case GLUT_DOWN: press = true; break;
		case GLUT_UP:   press = false; break;
	}

	if (button == GLUT_LEFT_BUTTON && press) {
		scene.setNewConePosition(pX, windowHeight - pY);
	}

	glutPostRedisplay();
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.update();

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	
	delete fullScreenTexturedQuad;
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	glutPostRedisplay();
}