/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_real_distribution;
using std::discrete_distribution;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 500;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i=0; i<num_particles; ++i){
    Particle part;
    part.x = dist_x(gen);
    part.y = dist_y(gen);
    part.theta = dist_theta(gen);
    part.weight = 1.0;
    particles.push_back(part);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  normal_distribution<double> noise_x(0.0, std_pos[0]);
  normal_distribution<double> noise_y(0.0, std_pos[1]);
  normal_distribution<double> noise_theta(0.0, std_pos[2]);

  for (int i=0; i<num_particles; ++i){
    particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + noise_x(gen);
    particles[i].y += velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + noise_y(gen);
    particles[i].theta += yaw_rate*delta_t + noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs>& observations,
                                     const Map &map_landmarks) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int p=0; p<num_particles; ++p){
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    for (int o=0; o<observations.size(); ++o){
      double map_frame_x = particles[p].x + cos(particles[p].theta)*observations[o].x - sin(particles[p].theta)*observations[o].y;
      double map_frame_y = particles[p].y + sin(particles[p].theta)*observations[o].x + cos(particles[p].theta)*observations[o].y;

      //check which landmark is closest to this observation
      double closest_landmark_distance = std::numeric_limits<double>::infinity();
      int closest_landmark_id;
      for (int l=0; l<map_landmarks.landmark_list.size(); ++l){
        double distance = dist(map_frame_x, map_frame_y, map_landmarks.landmark_list[l].x_f, map_landmarks.landmark_list[l].y_f);
        if (distance < closest_landmark_distance){
          closest_landmark_distance = distance;
          closest_landmark_id = map_landmarks.landmark_list[l].id_i;
        }
      }
      
      associations.push_back(closest_landmark_id);
      sense_x.push_back(map_frame_x);
      sense_y.push_back(map_frame_y);
    }
    SetAssociations(particles[p], associations, sense_x, sense_y);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  dataAssociation(observations, map_landmarks);

  for (int p=0; p<num_particles; ++p){
    for (int a=0; a<particles[p].associations.size(); ++a){
      double landmark_x = map_landmarks.landmark_list[particles[p].associations[a]].x_f;
      double landmark_y = map_landmarks.landmark_list[particles[p].associations[a]].y_f;
      double sense_x = particles[p].sense_x[a];
      double sense_y = particles[p].sense_y[a];
      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];
      //std::cout << landmark_x << " " << landmark_y << " " << sense_x << " " << sense_y << " " << sigma_x << " " << sigma_y << std::endl;
      particles[p].weight *= exp(-pow(landmark_x-sense_x,2)/(2*pow(sigma_x,2)))*exp(-pow(landmark_y-sense_y,2)/(2*pow(sigma_y,2)))/(2*M_PI*sigma_x*sigma_y);
    }
  }
  
  weights.clear();
  for (int p=0; p<num_particles; ++p){
    weights.push_back(particles[p].weight);
  }

  /** Normalize weights to avoid running into floating point precision errors, this 
   *  constant factor does not affect resampling
   */
  double weights_sum = std::accumulate(weights.begin(), weights.end(), 0);
  for (int i=0; i<weights.size(); ++i){
    weights[i] *= weights_sum;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampled_particles;

  double max_weight = *std::max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> dist_beta(0, 2*max_weight);
  discrete_distribution<int> dist_ind(0, num_particles);

  double beta = 0;
  int ind = dist_ind(gen);

  for (int p=0; p<num_particles; ++p){
    beta += dist_beta(gen);
    while (weights[ind] < beta){
      beta -= weights[ind];
      ind = (ind + 1) % num_particles;
    }
    resampled_particles.push_back(particles[ind]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}