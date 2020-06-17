#include "student_code.h"
#include "mutablePriorityQueue.h"
#include <numeric>
#include <queue>
#include <limits>

using namespace std;

namespace CGL
{
  /**
   * Evaluates one step of the de Casteljau's algorithm using the given points and
   * the scalar parameter t (class member).
   *
   * @param points A vector of points in 2D
   * @return A vector containing intermediate points or the final interpolated vector
   */
  std::vector<Vector2D> BezierCurve::evaluateStep(std::vector<Vector2D> const &points)
  { 
    // TODO Part 1.
    std::vector<Vector2D> new_vectors(points.size()-1);
    float t = BezierCurve::t;
    for (int i = 0; i < points.size() - 1; i++) {
        new_vectors[i] = (1-t)*points[i] + t*points[i+1];
    }
    return new_vectors;
  }

  /**
   * Evaluates one step of the de Casteljau's algorithm using the given points and
   * the scalar parameter t (function parameter).
   *
   * @param points    A vector of points in 3D
   * @param t         Scalar interpolation parameter
   * @return A vector containing intermediate points or the final interpolated vector
   */
  std::vector<Vector3D> BezierPatch::evaluateStep(std::vector<Vector3D> const &points, double t) const
  {
    // TODO Part 2.
    std::vector<Vector3D> new_vectors(points.size()-1);
    for (int i = 0; i < points.size() - 1; i++) {
        new_vectors[i] = (1-t)*points[i] + t*points[i+1];;
    }
    return new_vectors;
  }

  /**
   * Fully evaluates de Casteljau's algorithm for a vector of points at scalar parameter t
   *
   * @param points    A vector of points in 3D
   * @param t         Scalar interpolation parameter
   * @return Final interpolated vector
   */
  Vector3D BezierPatch::evaluate1D(std::vector<Vector3D> const &points, double t) const
  {
    // TODO Part 2.
    std::vector<Vector3D> intermediate = points;
    while (intermediate.size() > 1) {
        intermediate = evaluateStep(intermediate, t);
    }
    return intermediate[0];
  }

  /**
   * Evaluates the Bezier patch at parameter (u, v)
   *
   * @param u         Scalar interpolation parameter
   * @param v         Scalar interpolation parameter (along the other axis)
   * @return Final interpolated vector
   */
  Vector3D BezierPatch::evaluate(double u, double v) const 
  {  
    // TODO Part 2.
    std::vector< std::vector<Vector3D> > const controlPoints = BezierPatch::controlPoints;
    double n = controlPoints.size();
    std::vector<Vector3D> intermediate(n);
    for (int i = 0; i < n; i++) {
        std::vector<Vector3D> row = controlPoints[i];
        intermediate[i] = evaluate1D(controlPoints[i], u);
    }

    return evaluate1D(intermediate, v);
  }

  Vector3D Vertex::normal( void ) const
  {
    // TODO Part 3.
    // Returns an approximate unit normal at this vertex, computed by
    // taking the area-weighted average of the normals of neighboring
    // triangles, then normalizing.
    Vector3D position = Vertex::position;
    HalfedgeCIter h = Vertex::halfedge();      // get the outgoing half-edge of the vertex
    VertexCIter v = h->vertex();
    std::vector<Vector3D> neighbors;
    if (!Vertex::isBoundary()) {
        do {
            HalfedgeCIter h_twin = h->twin();
            Vector3D temp = h->face()->normal();
            VertexCIter v = h_twin->vertex();
            neighbors.push_back(temp);
            h = h_twin->next();

        } while (h != v->halfedge());

        Vector3D sum = neighbors[0];
        for (int i = 1; i < neighbors.size(); i++) {
            sum += neighbors[i];
        }
        return sum / sum.norm();
    } else {
        return Vector3D();
    }
  }

  EdgeIter HalfedgeMesh::flipEdge( EdgeIter e0 )
  {
    // TODO Part 4.
    // This method should flip the given edge and return an iterator to the flipped edge.
    if (!e0->getEdge()->isBoundary()) {
        HalfedgeIter h0 = e0->halfedge();
        HalfedgeIter h1 = h0->next();
        HalfedgeIter h2 = h1->next();
        HalfedgeIter h3 = h0->twin();
        HalfedgeIter h4 = h3->next();
        HalfedgeIter h5 = h4->next();
        HalfedgeIter h6 = h1->twin();
        HalfedgeIter h7 = h2->twin();
        HalfedgeIter h8 = h4->twin();
        HalfedgeIter h9 = h5->twin();

        VertexIter v0 = h0->vertex();
        VertexIter v1 = h3->vertex();
        VertexIter v2 = h2->vertex();
        VertexIter v3 = h5->vertex();

        EdgeIter e1 = h1->edge();
        EdgeIter e2 = h2->edge();
        EdgeIter e3 = h4->edge();
        EdgeIter e4 = h5->edge();

        FaceIter f0 = h0->face();
        FaceIter f1 = h3->face();

        h0->next() = h1;
        h0->twin() = h3;
        h0->vertex() = v3;
        h0->edge() = e0;
        h0->face() = f0;

        h1->next() = h2;
        h1->twin() = h7;
        h1->vertex() = v2;
        h1->edge() = e2;
        h1->face() = f0;

        h2->next() = h0;
        h2->twin() = h8;
        h2->vertex() = v0;
        h2->edge() = e3;
        h2->face() = f0;

        h3->next() = h4;
        h3->twin() = h0;
        h3->vertex() = v2;
        h3->edge() = e0;
        h3->face() = f1;

        h4->next() = h5;
        h4->twin() = h9;
        h4->vertex() = v3;
        h4->edge() = e4;
        h4->face() = f1;

        h5->next() = h3;
        h5->twin() = h6;
        h5->vertex() = v1;
        h5->edge() = e1;
        h5->face() = f1;

        h6->next() = h6->next();
        h6->twin() = h5;
        h6->vertex() = v2;
        h6->edge() = e1;
        h6->face() = h6->face();

        h7->next() = h7->next();
        h7->twin() = h1;
        h7->vertex() = v0;
        h7->edge() = e2;
        h7->face() = h7->face();

        h8->next() = h8->next();
        h8->twin() = h2;
        h8->vertex() = v3;
        h8->edge() = e3;
        h8->face() = h8->face();

        h9->next() = h9->next();
        h9->twin() = h4;
        h9->vertex() = v1;
        h9->edge() = e4;
        h9->face() = h9->face() ;

        v0->halfedge() = h2;
        v1->halfedge() = h5;
        v2->halfedge() = h3;
        v3->halfedge() = h0;

        e0->halfedge() = h0;
        e1->halfedge() = h5;
        e2->halfedge() = h1;
        e3->halfedge() = h2;
        e4->halfedge() = h4;

        f0->halfedge() = h0;
        f1->halfedge() = h3;

        return e0;
    }
    return e0;
  }

  VertexIter HalfedgeMesh::splitEdge( EdgeIter e0 )
  {
    // TODO Part 5.
    // This method should split the given edge and return an iterator to the newly inserted vertex.
    // The halfedge of this vertex should point along the edge that was split, rather than the new edges.

      if (e0->getEdge()->isBoundary()) {
          return e0->halfedge()->vertex();
      }
          //step 1: collect all the values
          HalfedgeIter h0 = e0->halfedge();
          HalfedgeIter h1 = h0->next();
          HalfedgeIter h2 = h1->next();
          HalfedgeIter h3 = h0->twin();
          HalfedgeIter h4 = h3->next();
          HalfedgeIter h5 = h4->next();

          HalfedgeIter h6 = h1->twin();
          HalfedgeIter h7 = h2->twin();
          HalfedgeIter h8 = h4->twin();
          HalfedgeIter h9 = h5->twin();

          VertexIter v0 = h0->vertex();
          VertexIter v1 = h3->vertex();
          VertexIter v2 = h2->vertex();
          VertexIter v3 = h5->vertex();

          EdgeIter e1 = h1->edge();
          EdgeIter e2 = h2->edge();
          EdgeIter e3 = h4->edge();
          EdgeIter e4 = h5->edge();

          FaceIter f1 = h0->face();
          FaceIter f0 = h3->face();

          //step 2: make new values
          VertexIter v4 = HalfedgeMesh::newVertex();
          HalfedgeIter h10 = HalfedgeMesh::newHalfedge();
          HalfedgeIter h11 = HalfedgeMesh::newHalfedge();
          HalfedgeIter h12 = HalfedgeMesh::newHalfedge();
          HalfedgeIter h13 = HalfedgeMesh::newHalfedge();
          HalfedgeIter h14 = HalfedgeMesh::newHalfedge();
          HalfedgeIter h15 = HalfedgeMesh::newHalfedge();
          FaceIter f2 = HalfedgeMesh::newFace();
          FaceIter f3 = HalfedgeMesh::newFace();

          v4->position = 0.5 * (v0->position + v1->position);

          v4->halfedge() = h0;
          EdgeIter e6 = HalfedgeMesh::newEdge();
          e6->halfedge() = h10;
          EdgeIter e7 = HalfedgeMesh::newEdge();
          e7 -> halfedge() = h12;
          EdgeIter e5 = HalfedgeMesh::newEdge();
          e5 -> halfedge() = h15;

          //step 3: reassign all the values
          //start with the half-edges
          h0->setNeighbors(h1, h3, v4, e0, f1);
          h1->setNeighbors(h12, h6, v1, e1, f1);
          h2->setNeighbors(h15, h7, v2, e2, f3);
          h3->setNeighbors(h10, h0, v1, e0, f0);
          h4->setNeighbors(h11, h8, v0, e3, f2);
          h5->setNeighbors(h3, h9, v3, e4, f0);
          h6->setNeighbors(h6->next(), h1, v2, e1, h6->face());
          h7->setNeighbors(h7->next(), h2, v0, e2, h7->face());
          h8->setNeighbors(h8->next(), h4, v3, e3, h8->face());
          h9->setNeighbors(h9->next(), h5, v1, e4, h9->face());
          h10->setNeighbors(h5, h11, v4, e6, f0);
          h11->setNeighbors(h14, h10, v3, e6, f2);
          h12->setNeighbors(h0, h13, v2, e7, f1);
          h13->setNeighbors(h2, h12, v4, e7, f3);
          h14->setNeighbors(h4, h15, v4, e5, f2);
          h15->setNeighbors(h13, h14, v0, e5, f3);

          //now the vertices
          v0->halfedge() = h15;
          v1->halfedge() = h3;
          v2->halfedge() = h2;
          v3->halfedge() = h5;
          v4->halfedge() = h0;


          //now the edges
          e0->halfedge() = h0;
          e1->halfedge() = h1;
          e2->halfedge() = h2;
          e3->halfedge() = h4;
          e4->halfedge() = h5;
          e5->halfedge() = h15;
          e6->halfedge() = h10;
          e7->halfedge() = h13;

          //now the faces
          f0->halfedge() = h3;
          f1->halfedge() = h0;
          f2->halfedge() = h14;
          f3->halfedge() = h15;
          return v4;
  }


  //** Final project begin.
  //************************************************************************//
  //************************************************************************//
  //************************************************************************//

  /**written for cs184 final project
  * this is actually collapse edge, but right now I named it splitedge for debugging purposes
  * merges the vertices on each end of a selected edge and returns the merged vertex
  * Parameters: h4 is the halfedge from v0 -> v2
  * comment made by @Katniss
  */

  VertexIter HalfedgeMesh::collapseEdge(EdgeIter e) {
      // check for edge cases
      if (e->isBoundary()) {
          return e->halfedge()->vertex();
      }

      //assign all the halfedges
      HalfedgeIter h4 = e->halfedge();
      HalfedgeIter h5 = h4->twin();
      HalfedgeIter h0 = h5->next();
      HalfedgeIter h6 = h0->twin();
      HalfedgeIter h1 = h0->next();
      HalfedgeIter h7 = h1->twin();
      HalfedgeIter h2 = h4->next();
      HalfedgeIter h3 = h2->next();
      HalfedgeIter h9 = h3->twin();
      HalfedgeIter h8 = h2->twin();

      //assign all the vertices
      VertexIter v0 = h0->vertex();
      VertexIter v1 = h1->vertex();
      VertexIter v2 = h2->vertex();
      VertexIter v3 = h3->vertex();

      //assign all the edges
      EdgeIter e1 = h1->edge();
      EdgeIter e2 = h2->edge();
      EdgeIter e3 = h3->edge();
      EdgeIter e4 = h0->edge();

      //assign all the faces
      FaceIter f0 = h0->face();
      FaceIter f1 = h2->face();

      //create new ones needed
      VertexIter v_opt = newVertex();
      EdgeIter e5 = newEdge();
      EdgeIter e6 = newEdge();

      // handle v_opt operation here
      // if either v0 or v2 are boundaries, just end here
      if( !v0->isBoundary() && !v2->isBoundary()) {
          reassignVopt(h4, v_opt);
          reassignVopt(h5, v_opt);
          v_opt->position = (v0->degree() * v0->position + v2->degree() * v2->position) / (v0->degree() + v2->degree());
          v_opt->halfedge() = h7;
      } else {
          v0->isBoundary() ? v_opt = v0 : v_opt = v2;
          return v_opt;
      }

      //handle remaining operations here
      h7->twin() = h6;
      h7->edge() = e6;
      h8->twin() = h9;
      h8->edge() = e5;
      h9->twin() = h8;
      h9->edge() = e5;
      h6->twin() = h7;
      h6->edge() = e6;
      v1->halfedge() = h6;
      e6->halfedge() = h7;
      v3->halfedge() = h8;
      e5->halfedge() = h9;

      //delete all the junk

      //halfedges
      deleteHalfedge(h4);
      deleteHalfedge(h2);
      deleteHalfedge(h3);
      deleteHalfedge(h5);
      deleteHalfedge(h0);
      deleteHalfedge(h1);

      //faces
      deleteFace(f1);
      deleteFace(f0);

      //edges
      deleteEdge(e);
      deleteEdge(e1);
      deleteEdge(e2);
      deleteEdge(e3);
      deleteEdge(e4);

      //vertices
      deleteVertex(v0);
      deleteVertex(v2);
      return v_opt;
  }

  class PairComparator {
    public:
        double operator() (const HalfedgeIter p1, const HalfedgeIter p2) {
            return p1->error > p2->error;
        }
    };

  void HalfedgeMesh::reassignVopt(HalfedgeIter h, VertexIter v_opt) {
        HalfedgeIter pointer = h;
        do {
            pointer->vertex() = v_opt;
            pointer = pointer->twin()->next();
        } while (pointer != h);
  }

  /** This is the main method of mesh simplification that checks whether two vertices are valid.
   * Takes in the mesh and iteratively collapses edges until the priority queue is empty or we
   * reach a minimum number of triangles (faces).
   * comment made by @Akshay
   */
  void MeshResampler::downsample(HalfedgeMesh& mesh) {
      // change the divisor
      int MIN_TRIANGLES = double(mesh.nFaces()) * 0.7;
      MIN_TRIANGLES = double(mesh.nFaces() - 10);
      priority_queue<HalfedgeIter, vector<HalfedgeIter>, PairComparator> pq;
      // fill PQ with all valid pairs
      /*
        for (VertexIter v1 = mesh.verticesBegin(); v1 != mesh.verticesEnd(); v1++) {
            quadricError(v1);
        }
      */

      // for (HalfedgeIter h = mesh.halfedgesBegin(); h != mesh.halfedgesEnd(); h++) {
      //    pq.insert(h);
      // }
      bool optimal;
      double multiplier = 0.5;
      double determinant_threshold = 1.0;

      HalfedgeIter h;
      double min_cost = std::numeric_limits<double>::max();
      optimal = false;
      for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++) {
          v_bar(e, optimal, determinant_threshold);
          if (e->halfedge()->error < min_cost) {
              min_cost = e->halfedge()->error;
              h = e->halfedge();
          }
      }

      int count = int((multiplier) * 0.5 * double(mesh.nFaces()));

      //mesh.nFaces() > MIN_TRIANGLES
      while (count != 1) {
          //HalfedgeIter toContract = pq.pop()->twin()->twin();
          //VertexIter v0 = toContract->vertex();
          //while (! pq.empty() && toContract->vertex()->isCollapsed) { toContract = pq.pop()->twin()->twin(); }
          //cout << toContract->getHalfedge() << " " << toContract->vertex()->degree() << "\n";
          min_cost = std::numeric_limits<double>::max();
          VertexIter v = mesh.collapseEdge(h->edge());
          /*quadricError(v0);
          HalfedgeIter h = v0->halfedge();
          do {
              pq.insert(h->getHalfedge());
              h = h->twin()->next();
          } while (h != v0->halfedge());*/

          recomputeEdgeCosts(v, optimal, determinant_threshold);
          // Computes the next minimum half-edge to collapse
          for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++) {
              if (e->halfedge()->error < min_cost) {
                  min_cost = e->halfedge()->error;
                  h = e->halfedge();
              }
          }
          count -= 1;
      }
  }



  /** This is a helper method that checks whether two vertices are valid.
   * Takes two vertexiters and returns boolean.
   * comment made by @Katniss
   */
  bool MeshResampler::isValidPair(VertexIter v0, VertexIter v1) {
      /* checks whether two vertices are a valid pair
      * 1. first is when the difference between the two vectors is less than the threshold
      */
      double t = 0.0; //using a threshold of t = 0 gives a simple edge contraction algorithm
      if ((v0->position - v1->position).norm() < t) {
          return true;
      }

      // 2. second is when the two vertices is an edge
      HalfedgeIter h = v0->halfedge();
      HalfedgeIter pointer = h;
      do {
          if (pointer->twin()->vertex() == v1) {
              return true;
          }
          pointer = pointer->twin()->next();
      } while(pointer != h);

      //if none of this works, return false
    return false;
  }

  /** This is a helper method that computers the error quadric matrix of a given vertex
   * Takes a vertex and returns a 4x4 matrix.
   * comment made by @Katniss
   */
  double quadricErrorHalfedge(VertexIter v1, VertexIter v2, HalfedgeIter h1, bool optimal, double determinant_threshold) {
      Vector3D v_bar = (v1->degree()*v1->position + v2->degree()*v2->position) / (v1->degree() + v2->degree());
      //for each vertex, we first collect all the faces that given vertex is connected to
      vector<FaceIter> faces;
      FaceIter v1_firstFace = v1->halfedge()->face();
      HalfedgeIter v1_pointer = v1->halfedge();
      Vector3D v1pos = v1->position;
      FaceIter v2_firstFace = v2->halfedge()->face();
      HalfedgeIter v2_pointer = v2->halfedge();
      Vector3D v2pos = v2->position;
      double error_v = 0.0;

      Matrix3x3 A_v1;
      A_v1[0] = Vector3D(0, 0, 0);
      A_v1[1] = Vector3D(0, 0, 0);
      A_v1[2] = Vector3D(0, 0, 0);
      double c_v1;
      Vector3D b_v1 = Vector3D(0, 0, 0);

      Matrix3x3 A_v2;
      A_v2[0] = Vector3D(0, 0, 0);
      A_v2[1] = Vector3D(0, 0, 0);
      A_v2[2] = Vector3D(0, 0, 0);
      double c_v2;
      Vector3D b_v2 = Vector3D(0, 0, 0);
      double error_v1;
      double error_v2;

      do {
          Vector3D face_normal = v1_pointer->face()->normal();
          double a_sq = face_normal.x * face_normal.x;
          double ab = face_normal.x * face_normal.y;
          double ac = face_normal.x * face_normal.z;
          double bc = face_normal.y * face_normal.z;
          double b_sq = face_normal.y * face_normal.y;
          double c_sq = face_normal.z * face_normal.z;
          A_v1[0] = A_v1[0] + Vector3D(a_sq, ab, ac);
          A_v1[1] = A_v1[1] + Vector3D(ab, b_sq, bc);
          A_v1[2] = A_v1[2] + Vector3D(ac, bc, c_sq);

          // Vector3D v2pos = v1_pointer->next()->vertex()->position;
          // Vector3D v3pos = v1_pointer->next()->next()->vertex()->position;
          // double centroid_x = (1.0 / 3.0) * (vpos.x + v2pos.x + v3pos.x);
          // double centroid_y = (1.0 / 3.0) * (vpos.y + v2pos.y + v3pos.y);
          // double centroid_z = (1.0 / 3.0) * (vpos.z + v2pos.z + v3pos.z);
          double d = (-1) * (face_normal.x * v1pos.x + face_normal.y * v1pos.y + face_normal.z * v1pos.z);
          b_v1 = b_v1 + d * face_normal;
          c_v1 = c_v1 + (d * d);
          double distance = (face_normal.x * v_bar.x + face_normal.y * v_bar.y + face_normal.z * v_bar.z + d);
          error_v += (distance * distance);
          error_v1 = dot(v_bar, (A_v1 * v_bar)) + 2 * dot(b_v1, v_bar) + c_v1;
          v1_pointer = v1_pointer->next()->next()->twin();
      } while(v1_pointer->face() != v1_firstFace);

      do {
          Vector3D face_normal = v2_pointer->face()->normal();
          double a_sq = face_normal.x * face_normal.x;
          double ab = face_normal.x * face_normal.y;
          double ac = face_normal.x * face_normal.z;
          double bc = face_normal.y * face_normal.z;
          double b_sq = face_normal.y * face_normal.y;
          double c_sq = face_normal.z * face_normal.z;
          A_v2[0] = A_v2[0] + Vector3D(a_sq, ab, ac);
          A_v2[1] = A_v2[1] + Vector3D(ab, b_sq, bc);
          A_v2[2] = A_v2[2] + Vector3D(ac, bc, c_sq);

          // Vector3D v2pos = v1_pointer->next()->vertex()->position;
          // Vector3D v3pos = v1_pointer->next()->next()->vertex()->position;
          // double centroid_x = (1.0 / 3.0) * (vpos.x + v2pos.x + v3pos.x);
          // double centroid_y = (1.0 / 3.0) * (vpos.y + v2pos.y + v3pos.y);
          // double centroid_z = (1.0 / 3.0) * (vpos.z + v2pos.z + v3pos.z);
          double d = (-1) * (face_normal.x * v2pos.x + face_normal.y * v2pos.y + face_normal.z * v2pos.z);
          b_v2 = b_v2 + d * face_normal;
          c_v2 = c_v2 + (d * d);
          double distance = (face_normal.x * v_bar.x + face_normal.y * v_bar.y + face_normal.z * v_bar.z + d);
          error_v += (distance * distance);
          error_v2 = dot(v_bar, (A_v2 * v_bar)) + 2 * dot(b_v2, v_bar) + c_v2;
          double error_total = error_v2 + error_v1;
          v2_pointer = v2_pointer->next()->next()->twin();
      } while(v2_pointer->face() != v2_firstFace);

      if (optimal) {
          Matrix3x3 A_vbar;
          A_vbar[0] = A_v1[0] + A_v2[0];
          A_vbar[1] = A_v1[1] + A_v2[1];
          A_vbar[2] = A_v1[2] + A_v2[2];
          Vector3D b_vbar = b_v1 + b_v2;
          double c_vbar = c_v1 + c_v2;
          if (A_vbar.det() > determinant_threshold) {
              Matrix3x3 A_inv = (-1) * A_vbar.inv();
              v_bar = A_inv * b_vbar;
              error_v = dot(b_vbar, v_bar) + c_vbar;
          }
      }
      h1->error = error_v;
      h1->v_bar = v_bar;
      // v->faces = a vector of all the faces that this vertex has
      return 0.0;
  }

  void MeshResampler::v_bar(EdgeIter e, bool optimal, double determinant_threshold) {
      auto h1 = e->halfedge();
      auto h2 = h1->twin();
      VertexIter v1 = h1->vertex();
      VertexIter v2 = h2->vertex();
      quadricErrorHalfedge(v1, v2, h1, optimal, determinant_threshold);
  }

    void MeshResampler::recomputeEdgeCosts(VertexIter v, bool optimal, double determinant_threshold) {
        HalfedgeIter h = v->halfedge();      // get the outgoing half-edge of the vertex
        do {
            HalfedgeIter h_twin = h->twin(); // get the opposite half-edge
            VertexIter v2 = h_twin->vertex(); // vertex is the 'source' of the half-edge, so
            quadricErrorHalfedge(v, v2, h->edge()->halfedge(), optimal, determinant_threshold);
            h = h_twin->next();
        } while(h != v->halfedge());

    }

  //** Final project end.
  //************************************************************************//
  //************************************************************************//
  //************************************************************************//


  void MeshResampler::upsample( HalfedgeMesh& mesh )
  {
    // TODO Part 6.
    // This routine should increase the number of triangles in the mesh using Loop subdivision.
      vector<EdgeIter> aftersplit;

      for (VertexIter v = mesh.verticesBegin(); v != mesh.verticesEnd(); v++) {
          v->isOld = true;
          double n = v->degree();
          double u = 0.0;
          if (n == 3.0) {
              u = 3.0/16.0;
          } else {
              u = 3.0/(8.0 * n);
          }
          Vector3D neighbor_sum = Vector3D(0.0, 0.0, 0.0);
          HalfedgeIter h = v->halfedge();
          HalfedgeIter start_h = h;
          do {
              neighbor_sum += h->next()->vertex()->position;
              h = h->next()->next()->twin();
          } while (h != start_h);

          double constant = (double) (1.0 - n * u);
          v->newPosition = constant * v->position + u * neighbor_sum;
      }

      for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++) {
          e->isOld = true;
          e->isBlack = true;
          VertexIter A = e->halfedge()->vertex();
          VertexIter B = e->halfedge()->twin()->vertex();
          VertexIter C = e->halfedge()->next()->next()->vertex();
          VertexIter D = e->halfedge()->twin()->next()->next()->vertex();
          e->newPosition = (3.0/8.0) * (A->position + B->position) + (1.0/8.0) * (C->position + D->position);
      }

      for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++) {
          if (e->isOld == true) {
              VertexIter v4 = mesh.splitEdge(e);
              HalfedgeIter h0 = v4->halfedge();
              EdgeIter e0 = h0->edge();
              EdgeIter e5 = h0->twin()->next()->twin()->next()->edge();
              v4->newPosition = e->newPosition;
              e0->isBlack=true;
              e5->isBlack=true;
          }
      }

      for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++) {
          aftersplit.push_back(e);
      }

      //flip all the edges that contain one original and one new
      //for all the edges, if it's not in the original edges:
      //if the edge's vertex is
      for (int i = 0; i < aftersplit.size(); i++) {
          EdgeIter e = aftersplit[i];
          if (e->isOld == false) {
              if (e->isBlack == false) {
                  HalfedgeIter h = e->halfedge();
                  HalfedgeIter t = h->twin();
                  if ((h->vertex()->isOld == true && t->vertex()->isOld == false)
                      || (h->vertex()->isOld == false && t->vertex()->isOld == true)) {
                      mesh.flipEdge(e);
                  }
              }
          }
      }

      for (VertexIter v = mesh.verticesBegin(); v != mesh.verticesEnd(); v++) {
          v->position = v->newPosition;
          v->isOld=false;
      }

      for (EdgeIter e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++) {
          e->isOld=false;
      }
  }
}