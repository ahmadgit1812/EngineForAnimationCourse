#include "BasicScene.h"
#include <read_triangle_mesh.h>
#include <utility>
#include "ObjLoader.h"
#include "IglMeshLoader.h"
#include "igl/read_triangle_mesh.cpp"
#include "igl/edge_flaps.h"

// #include "AutoMorphingModel.h"

using namespace cg3d;

void BasicScene::Init(float fov, int width, int height, float near, float far)
{
    camera = Camera::Create( "camera", fov, float(width) / height, near, far);

    AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
    auto daylight{std::make_shared<Material>("daylight", "shaders/cubemapShader")};
    daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
    auto background{Model::Create("background", Mesh::Cube(), daylight)};
    AddChild(background);
    background->Scale(120, Axis::XYZ);
    background->SetPickable(false);
    background->SetStatic();

    auto program = std::make_shared<Program>("shaders/basicShader");
    auto material{ std::make_shared<Material>("material", program)};
    material->AddTexture(0, "textures/box0.bmp", 2);

    auto sphere1Mesh{IglLoader::MeshFromFiles("sphere1_igl", "data/bunny.off")};
    auto sphere2Mesh{IglLoader::MeshFromFiles("sphere2_igl","data/bunny.off")};

    auto cubeMesh1 = Mesh::Cube();

    sphere1 = Model::Create( "sphere1",sphere1Mesh, material);
    sphere2 = Model::Create( "sphere2", sphere2Mesh, material);



    sphere1->showWireframe = true;
    sphere1->Translate({-3,0,0});
    sphere1->Scale(17);

    sphere2->showWireframe = true;
    sphere2->Translate({3,0,0});
    sphere2->Scale(17);


    camera->Translate(20, Axis::Z);
    root->AddChild(sphere1);
    root->AddChild(sphere2);


    sphere1->init_bounding_box(sphere1->model_tree.m_box);
    sphere2->init_bounding_box(sphere2->model_tree.m_box);


}



void BasicScene::Update(const Program& program, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, const Eigen::Matrix4f& model)
{
    Scene::Update(program, proj, view, model);
    program.SetUniform4f("lightColor", 1.0f, 1.0f, 1.0f, 0.5f);
    program.SetUniform4f("Kai", 1.0f, 1.0f, 1.0f, 1.0f);

    sphere1->translate_velocity();
    sphere2->translate_velocity();
    // Check if a collision occurred
    if (!is_collision_detected && sphere1->detect_collision(sphere2)) {

        sphere1->zero_velocity();
        sphere2->zero_velocity();

        is_collision_detected = true;
        std::cout << "col is detected" << std::endl;

    }

}