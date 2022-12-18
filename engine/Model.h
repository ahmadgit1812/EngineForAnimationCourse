#pragma once

#include <iostream>
#include "Mesh.h"
#include "Material.h"
#include "Movable.h"
#include "ViewerData.h"
#include "igl/AABB.h"

namespace cg3d
{

    class Model : virtual public Movable
    {
        friend class DrawVisitor;

    protected:
        Model(const std::string& file, std::shared_ptr<Material> material);
        Model(std::string name, const std::string& file, std::shared_ptr<Material> material);
        Model(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material);
        Model(std::string name, std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material);
        Model(std::string name, std::vector<std::shared_ptr<Mesh>> meshList, std::shared_ptr<Material> material);
        Model(const Model& other) = default; // important: doesn't add itself to the parent's children (object isn't constructed yet)
        Model(Model&&) = default; // important: doesn't add itself to the parent's children (object isn't constructed yet)
        Model& operator=(const Model& other) = default;

    public:
        template<typename... Args>
        static std::shared_ptr<Model> Create(Args&&... args) {
            return std::shared_ptr<Model>{new Model{std::forward<Args>(args)...}}; // NOLINT(modernize-make-shared)
        }

        ~Model() override = default;

        void Accept(Visitor* visitor) override { visitor->Visit(this); };

        std::shared_ptr<Material> material;
        bool isHidden = false;
        bool showFaces = true;
        bool showTextures = true;
        bool showWireframe = false;
        Eigen::Vector4f wireframeColor{0, 0, 0, 0};
        int meshIndex = 0;
        int meshMaxIndex = 0;


        igl::AABB<Eigen::MatrixXd, 3> model_tree;
        int BoundingBoxIndex=-1;
        bool first = true;




        float model_y_velocity=0.0, model_x_velocity=0.0;
        void translate_velocity();
        void zero_velocity();

        void init_bounding_box(Eigen::AlignedBox<double, 3>& alignedB);
        void init_collision_box(Eigen::AlignedBox<double, 3>& alignedB);
        void init_model_tree();
        bool detect_collision(std::shared_ptr<Model>& other_model);
        bool detect_collision(igl::AABB<Eigen::MatrixXd, 3>* model1_tree, igl::AABB<Eigen::MatrixXd, 3>* model2_tree);
        std::shared_ptr<Model> model_checking_for_collision= nullptr;





        bool OrientedBoundingBoxesIntersect(Eigen::AlignedBox<double, 3>& firstAlignedBox, Eigen::AlignedBox<double, 3>& secAlignedBox);
        void OBI_init(Eigen::AlignedBox<double, 3>& firstAlignedBox, Eigen::AlignedBox<double, 3>& secAlignedBox);

        inline std::shared_ptr<Mesh> GetMesh(int index = 0) const { return meshList[index]; }
        inline std::vector<std::shared_ptr<Mesh>> GetMeshList() const { return meshList; }
        void SetMeshList(std::vector<std::shared_ptr<Mesh>> _meshList);
        void UpdateDataAndDrawMeshes(const Program& program, bool _showFaces, bool bindTextures); // helper function

    private:

        std::shared_ptr<Mesh> build_from_aligned_box(Eigen::AlignedBox<double, 3>& alignedB);
        static void UpdateDataAndBindMesh(igl::opengl::ViewerData& viewerData, const Program& program); // helper function

        static std::vector<igl::opengl::ViewerData> CreateViewerData(const std::shared_ptr<Mesh>& mesh);
        std::vector<std::shared_ptr<Mesh>> meshList;

        std::vector<std::vector<igl::opengl::ViewerData>> viewerDataListPerMesh;


        //intersection members
        Eigen::MatrixXd A_33_matrix;
        Eigen::MatrixXd B_33_matrix;
        Eigen::MatrixXd C_33_matrix;

        Eigen::Vector3d A_0_col;
        Eigen::Vector3d A_1_col;
        Eigen::Vector3d A_2_col;
        Eigen::Vector3d B_0_col;
        Eigen::Vector3d B_1_col;
        Eigen::Vector3d B_2_col;

        Eigen::Affine3f scale_obj;

        Eigen::Vector4d first_cen;
        Eigen::Vector4d sec_cen;

        Eigen::Matrix4f curr_Trans;

        Eigen::Matrix4d model_trans_cast;
        Eigen::Matrix4d other_model_trans_cast;
        Eigen::Vector4d D_2_op;
        Eigen::Vector4d D_1_op;
        Eigen::Vector4d D_4_cord;
        Eigen::Vector3d D_3_vector;



        Eigen::Vector3d a;
        Eigen::Vector3d b;

        Eigen::Vector3d firstAlignedBoxCenter;
        Eigen::Vector3d secAlignedBoxCenter;

        float R_temp_val;
        float R0_temp_val, R1_temp_val;



        // TODO: TAL: handle the colors...
        Eigen::RowVector4f ambient = Eigen::RowVector4f(1.0, 1.0, 1.0, 1.0);
        Eigen::RowVector4f diffuse = Eigen::RowVector4f(1.0, 1.0, 1.0, 1.0);
        Eigen::RowVector4f specular = Eigen::RowVector4f(1.0, 1.0, 1.0, 1.0);



    };

} // namespace cg3d
