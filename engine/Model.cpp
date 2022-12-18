#include "Model.h"
#include "Visitor.h"
#include "ViewerData.h"
#include "Movable.h"
#include "ObjLoader.h"
#include <filesystem>
#include <utility>
#include "igl/per_vertex_normals.h"

namespace cg3d
{

    namespace fs = std::filesystem;

    Model::Model(const std::string& file, std::shared_ptr<Material> material)
            : Model{fs::path(file).filename().stem().generic_string(), file, std::move(material)} {
        Model::init_model_tree();

    }

    Model::Model(std::string name, const std::string& file, std::shared_ptr<Material> material)
            : Model{*ObjLoader::ModelFromObj(std::move(name), file, std::move(material))} {
        Model::init_model_tree();
    }

    Model::Model(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material)
            : Model{mesh->name + "_model", std::move(mesh), std::move(material)} {
        Model::init_model_tree();
    }

    Model::Model(std::string name, std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material)
            : Model{std::move(name), std::vector<std::shared_ptr<Mesh>>{{std::move(mesh)}}, std::move(material)} {
        Model::init_model_tree();
    }

    Model::Model(std::string name, std::vector<std::shared_ptr<Mesh>> meshList, std::shared_ptr<Material> material)
            : Movable(std::move(name)), material(std::move(material))
    {
        SetMeshList(std::move(meshList));
        Model::init_model_tree();
    }

    std::vector<igl::opengl::ViewerData> Model::CreateViewerData(const std::shared_ptr<Mesh>& mesh)
    {
        std::vector<igl::opengl::ViewerData> dataList;

        for (auto& meshData: mesh->data) {
            igl::opengl::ViewerData viewerData;
            viewerData.set_mesh(meshData.vertices, meshData.faces);
            viewerData.set_uv(meshData.textureCoords);
            viewerData.set_normals(meshData.vertexNormals);
            viewerData.line_width = 1.0f;
            viewerData.uniform_colors(Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Vector3d(1.0, 1.0, 1.0)); // todo: implement colors
            viewerData.compute_normals(); // todo: implement (this overwrites both face and vertex normals even if either is already present)
            if (viewerData.V_uv.rows() == 0)
                viewerData.grid_texture();
            viewerData.is_visible = 1;
            viewerData.show_overlay = 0;
            dataList.emplace_back(std::move(viewerData));
        }

        return dataList;
    }

    void Model::UpdateDataAndBindMesh(igl::opengl::ViewerData& viewerData, const Program& program)
    {
        viewerData.dirty = igl::opengl::MeshGL::DIRTY_NONE;
        viewerData.updateGL(viewerData, viewerData.invert_normals, viewerData.meshgl);
        viewerData.meshgl.shader_mesh = program.GetHandle();
        viewerData.meshgl.bind_mesh();
    }

    void Model::UpdateDataAndDrawMeshes(const Program& program, bool _showFaces, bool bindTextures)
    {

        for(int i=0; i<viewerDataListPerMesh.size(); i++){
            auto& viewerData = viewerDataListPerMesh[i][std::min(meshIndex, int(viewerDataListPerMesh[i].size() - 1))];
            UpdateDataAndBindMesh(viewerData, program);
            if (bindTextures) material->BindTextures();
            if(i != BoundingBoxIndex)
                viewerData.meshgl.draw_mesh(_showFaces);
            else
                viewerData.meshgl.draw_mesh(false);

        }

    }

    void Model::SetMeshList(std::vector<std::shared_ptr<Mesh>> _meshList)
    {
        meshList = std::move(_meshList);
        viewerDataListPerMesh.clear();
        for (auto& mesh: meshList) {
            viewerDataListPerMesh.emplace_back(CreateViewerData(mesh));
        }
    }

    void Model::translate_velocity(){
        Translate({model_x_velocity, model_y_velocity,0});
    }

    void Model::zero_velocity(){
        model_x_velocity=0.0;
        model_y_velocity=0.0;
    }


    void Model::init_model_tree(){
        model_tree.init(GetMesh()->data[0].vertices,GetMesh()->data[0].faces);
    }

    std::shared_ptr<Mesh> Model::build_from_aligned_box(Eigen::AlignedBox<double, 3>& alignedB){
        Eigen::MatrixXd V = Eigen::MatrixXd::Ones(8,3);

        V.row(0) = alignedB.corner(alignedB.BottomRightFloor);
        V.row(1) = alignedB.corner(alignedB.TopRightFloor);
        V.row(2) = alignedB.corner(alignedB.TopRightCeil);
        V.row(3) = alignedB.corner(alignedB.BottomRightCeil);
        V.row(4) = alignedB.corner(alignedB.BottomLeftCeil);
        V.row(5) = alignedB.corner(alignedB.BottomLeftFloor);
        V.row(6) = alignedB.corner(alignedB.TopLeftFloor);
        V.row(7) = alignedB.corner(alignedB.TopLeftCeil);


        Eigen::MatrixXi F = Eigen::MatrixXi::Ones(12,3);

        //right
        F.row(0) = Eigen::RowVector3i(1, 2, 3);
        F.row(1) = Eigen::RowVector3i(1, 0, 3);

        //bottom
        F.row(2) = Eigen::RowVector3i(5, 0, 3);
        F.row(3) = Eigen::RowVector3i(5, 4, 3);

        //left
        F.row(4) = Eigen::RowVector3i(4, 5, 6);
        F.row(5) = Eigen::RowVector3i(4, 7, 6);

        //top
        F.row(6) = Eigen::RowVector3i(2, 7, 6);
        F.row(7) = Eigen::RowVector3i(2, 1, 6);

        //front
        F.row(8) = Eigen::RowVector3i(4, 7, 2);
        F.row(9) = Eigen::RowVector3i(4, 3, 2);

        //back
        F.row(10) = Eigen::RowVector3i(5,0 , 1);
        F.row(11) = Eigen::RowVector3i(5, 6, 1);

        Eigen::MatrixXd box_vertex_normals;
        igl::per_vertex_normals(V, F, box_vertex_normals);

        return std::make_shared<Mesh>("", V, F, box_vertex_normals, Eigen::MatrixXd());
    }




    void Model::init_bounding_box(Eigen::AlignedBox<double, 3>& alignedB){
        showWireframe= true;
        BoundingBoxIndex = GetMeshList().size();
        meshList.push_back(build_from_aligned_box(alignedB));
        SetMeshList(GetMeshList());
    }



    void Model::init_collision_box(Eigen::AlignedBox<double, 3> &alignedB){

        std::shared_ptr<Model> collision_box = Model::Create(build_from_aligned_box(alignedB), material);
        collision_box->showWireframe = true;
        collision_box->wireframeColor = Eigen::Vector4f(0.5,0.7,0.6,1);
        AddChild(collision_box);
    }

    bool Model::detect_collision(std::shared_ptr<Model>& other_model){
        model_checking_for_collision = other_model;
        bool return_val = detect_collision(&model_tree, &other_model->model_tree);
        model_checking_for_collision = nullptr;
        return return_val;
    }


    bool Model::detect_collision(igl::AABB<Eigen::MatrixXd, 3>* model1_tree, igl::AABB<Eigen::MatrixXd, 3>* model2_tree){

        if (OrientedBoundingBoxesIntersect(model1_tree->m_box, model2_tree->m_box) == false){

            return false;

        }
        else if (model2_tree->is_leaf() && model1_tree->is_leaf() == false) {

            return (model1_tree->m_left != nullptr ? detect_collision(model1_tree->m_left, model2_tree) : false) ||
                   (model1_tree->m_right != nullptr ? detect_collision(model1_tree->m_right, model2_tree) : false);

        }
        else if (model1_tree->is_leaf() && model2_tree->is_leaf() == false) {

            return (model2_tree->m_left != nullptr ? detect_collision(model1_tree, model2_tree->m_left) : false) ||
                   (model2_tree->m_right != nullptr ? detect_collision(model1_tree, model2_tree->m_right) : false);

        }
        else if(model1_tree->is_leaf() == false && model2_tree->is_leaf() == false) {

            return
                    ((model1_tree->m_left != nullptr && model2_tree->m_right != nullptr) ? detect_collision(
                            model1_tree->m_left, model2_tree->m_right) : false) ||

                    ((model1_tree->m_left != nullptr && model2_tree->m_left != nullptr) ? detect_collision(
                            model1_tree->m_left, model2_tree->m_left) : false) ||

                    ((model1_tree->m_right != nullptr && model2_tree->m_right != nullptr) ? detect_collision(
                            model1_tree->m_right, model2_tree->m_right) : false) ||

                    ((model1_tree->m_right != nullptr && model2_tree->m_left != nullptr) ? detect_collision(
                            model1_tree->m_right, model2_tree->m_left) : false);

        }else{
            init_collision_box(model1_tree->m_box);
            model_checking_for_collision->init_collision_box(model2_tree->m_box);
            return true;
        }
    }

    void Model::OBI_init(Eigen::AlignedBox<double, 3>& firstAlignedBox, Eigen::AlignedBox<double, 3>& secAlignedBox){

        A_33_matrix = GetRotation().cast<double>();
        B_33_matrix = model_checking_for_collision->GetRotation().cast<double>();
        C_33_matrix = A_33_matrix.transpose() * B_33_matrix;

        // A and B columns init
        A_0_col << A_33_matrix(0, 0) , A_33_matrix(1, 0) ,A_33_matrix(2, 0);
        A_1_col << A_33_matrix(0, 1) , A_33_matrix(1, 1) ,A_33_matrix(2, 1);
        A_2_col << A_33_matrix(0, 2) , A_33_matrix(1, 2) ,A_33_matrix(2, 2);

        B_0_col << B_33_matrix(0, 0) , B_33_matrix(1, 0) ,B_33_matrix(2, 0);
        B_1_col << B_33_matrix(0, 1) , B_33_matrix(1, 1) ,B_33_matrix(2, 1);
        B_2_col << B_33_matrix(0, 2) , B_33_matrix(1, 2) ,B_33_matrix(2, 2);


        curr_Trans = GetTransform();
        scale_obj = GetScaling(curr_Trans);
        a = scale_obj(0, 0)*(firstAlignedBox.sizes() / 2);
        b = scale_obj(0, 0)*(secAlignedBox.sizes() / 2);

        firstAlignedBoxCenter = firstAlignedBox.center();
        secAlignedBoxCenter = secAlignedBox.center();
        first_cen[0] = firstAlignedBoxCenter[0];
        first_cen[1] = firstAlignedBoxCenter[1];
        first_cen[2] = firstAlignedBoxCenter[2];
        first_cen[3] = 1;

        sec_cen[0] = secAlignedBoxCenter[0];
        sec_cen[1] = secAlignedBoxCenter[1];
        sec_cen[2] = secAlignedBoxCenter[2];
        sec_cen[3] = 1;

        // Matrix D_3_vector
        model_trans_cast = GetTransform().cast<double>();
        other_model_trans_cast = model_checking_for_collision->GetTransform().cast<double>();
        D_2_op = model_trans_cast * first_cen;
        D_1_op = other_model_trans_cast * sec_cen;
        D_4_cord = D_1_op - D_2_op;
        D_3_vector = Eigen::Vector3d(D_4_cord(0), D_4_cord(1), D_4_cord(2));
    }


    bool Model::OrientedBoundingBoxesIntersect(Eigen::AlignedBox<double, 3>& firstAlignedBox, Eigen::AlignedBox<double, 3>& secAlignedBox)
    {

        OBI_init(firstAlignedBox, secAlignedBox);


        // first cond
        R1_temp_val = b(0) * abs(C_33_matrix(0,0));
        R1_temp_val += (b(1) * abs(C_33_matrix(0,1)));
        R1_temp_val += (b(2) * abs(C_33_matrix(0,2)));

        R_temp_val = abs(A_0_col.transpose() * D_3_vector);
        if(R_temp_val > a(0) + R1_temp_val )
            return false;

        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // sec cond
        R1_temp_val = b(0) * abs(C_33_matrix(1,0));
        R1_temp_val += (b(1) * abs(C_33_matrix(1,1)));
        R1_temp_val += (b(2) * abs(C_33_matrix(1,2)));

        R_temp_val = abs(A_1_col.transpose() * D_3_vector);
        if(R_temp_val > a(1) + R1_temp_val )
            return false;

        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // third cond
        R1_temp_val = b(0) * abs(C_33_matrix(2,0));
        R1_temp_val += (b(1) * abs(C_33_matrix(2,1)));
        R1_temp_val += (b(2) * abs(C_33_matrix(2,2)));

        R_temp_val = abs(A_2_col.transpose() * D_3_vector);
        if(R_temp_val > a(2) + R1_temp_val )
            return false;


        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // forth cond
        R0_temp_val = a(0) * abs(C_33_matrix(0,0));
        R0_temp_val += (a(1) * abs(C_33_matrix(1,0)));
        R0_temp_val += (a(2) * abs(C_33_matrix(2,0)));


        R_temp_val = abs(B_0_col.transpose() * D_3_vector);
        if(R_temp_val > R0_temp_val + b(0) )
            return false;

        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 5 cond
        R0_temp_val = a(0) * abs(C_33_matrix(0,1));
        R0_temp_val += (a(1) * abs(C_33_matrix(1,1)));
        R0_temp_val += (a(2) * abs(C_33_matrix(2,1)));


        R_temp_val = abs(B_1_col.transpose() * D_3_vector);
        if(R_temp_val > R0_temp_val + b(1) )
            return false;

        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 6 cond
        R0_temp_val = a(0) * abs(C_33_matrix(0,2));
        R0_temp_val += (a(1) * abs(C_33_matrix(1,2)));
        R0_temp_val += (a(2) * abs(C_33_matrix(2,2)));


        R_temp_val = abs(B_2_col.transpose() * D_3_vector);
        if(R_temp_val > R0_temp_val + b(2) )
            return false;

        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 00
        R0_temp_val = a(1) * abs(C_33_matrix(2,0));
        R0_temp_val += (a(2) * abs(C_33_matrix(1,0)));


        R1_temp_val = b(1) * abs(C_33_matrix(0,2));
        R1_temp_val += (b(2) * abs(C_33_matrix(0,1)));


        R_temp_val = C_33_matrix(1,0) * A_2_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(2,0) * A_1_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;



        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 01
        R0_temp_val = a(1) * abs(C_33_matrix(2,1));
        R0_temp_val += (a(2) * abs(C_33_matrix(1,1)));


        R1_temp_val = b(0) * abs(C_33_matrix(0,2));
        R1_temp_val += (b(2) * abs(C_33_matrix(0,0)));


        R_temp_val = C_33_matrix(1,1) * A_2_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(2,1) * A_1_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;


        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 02
        R0_temp_val = a(1) * abs(C_33_matrix(2,2));
        R0_temp_val += (a(2) * abs(C_33_matrix(1,2)));


        R1_temp_val = b(0) * abs(C_33_matrix(0,1));
        R1_temp_val += (b(1) * abs(C_33_matrix(0,0)));


        R_temp_val = C_33_matrix(1,2) * A_2_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(2,2) * A_1_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;

        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 10
        R0_temp_val = a(0) * abs(C_33_matrix(2,0));
        R0_temp_val += (a(2) * abs(C_33_matrix(0,0)));


        R1_temp_val = b(1) * abs(C_33_matrix(1,2));
        R1_temp_val += (b(2) * abs(C_33_matrix(1,1)));


        R_temp_val = C_33_matrix(2,0) * A_0_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(0,0) * A_2_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;


        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 11
        R0_temp_val = a(0) * abs(C_33_matrix(2,1));
        R0_temp_val += (a(2) * abs(C_33_matrix(0,1)));


        R1_temp_val = b(0) * abs(C_33_matrix(1,2));
        R1_temp_val += (b(2) * abs(C_33_matrix(1,0)));


        R_temp_val = C_33_matrix(2,1) * A_0_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(0,1) * A_2_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;


        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 12
        R0_temp_val = a(0) * abs(C_33_matrix(2,2));
        R0_temp_val += (a(2) * abs(C_33_matrix(0,2)));


        R1_temp_val = b(0) * abs(C_33_matrix(1,1));
        R1_temp_val += (b(1) * abs(C_33_matrix(1,0)));


        R_temp_val = C_33_matrix(2,2) * A_0_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(0,2) * A_2_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;


        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 20
        R0_temp_val = a(0) * abs(C_33_matrix(1,0));
        R0_temp_val += (a(1) * abs(C_33_matrix(0,0)));


        R1_temp_val = b(1) * abs(C_33_matrix(2,2));
        R1_temp_val += (b(2) * abs(C_33_matrix(2,1)));


        R_temp_val = C_33_matrix(0,0) * A_1_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(1,0) * A_0_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;


        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 21
        R0_temp_val = a(0) * abs(C_33_matrix(1,1));
        R0_temp_val += (a(1) * abs(C_33_matrix(0,1)));


        R1_temp_val = b(0) * abs(C_33_matrix(2,2));
        R1_temp_val += (b(2) * abs(C_33_matrix(2,0)));


        R_temp_val = C_33_matrix(0,1) * A_1_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(1,1) * A_0_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;
        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 22
        R0_temp_val = a(0) * abs(C_33_matrix(1,2));
        R0_temp_val += (a(1) * abs(C_33_matrix(0,2)));


        R1_temp_val = b(0) * abs(C_33_matrix(2,1));
        R1_temp_val += (b(1) * abs(C_33_matrix(2,0)));


        R_temp_val = C_33_matrix(0,2) * A_1_col.transpose() * D_3_vector;
        R_temp_val =  abs(R_temp_val - (C_33_matrix(1,2) * A_0_col.transpose() * D_3_vector));

        if(R_temp_val > R0_temp_val + R1_temp_val)
            return false;


        return true;
    }






} // namespace cg3d
