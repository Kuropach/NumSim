#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "./json.hpp"
#include <algorithm>
#include <functional>
#include <cmath>

using json = nlohmann::json;
using PendulumFunc = std::vector<double>(*)(std::vector<double>, double, std::vector<double>);

std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

std::vector<double> operator*(double scalar, const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = scalar * vec[i];
    }
    return result;
}

std::vector<double> MathPendulum(std::vector<double> params, double x, std::vector<double> y){
    double w = params[0];
    std::vector<double> f = {y[1], -w*w*y[0]};
    return f;
}

std::vector<double> PhysPendulum(std::vector<double> params, double x, std::vector<double> y){
    double w = params[0];
    std::vector<double> f = {y[1], -w*w*std::sin(y[0])};
    return f;
}

std::vector<double> DampedMathPendulum(std::vector<double> params, double x, std::vector<double> y){
    double w = params[0];
    double gamma = params[1];
    std::vector<double> f = {y[1], -w*w*y[0] - gamma*y[1]};
    return f;
}

std::vector<double> DampedPhysPendulum(std::vector<double> params, double x, std::vector<double> y){
    double w = params[0];
    double gamma = params[1];
    std::vector<double> f = {y[1], -w*w*std::sin(y[0]) - gamma*y[1]};
    return f;
}

std::vector<double> ForcedDampedMathPendulum(std::vector<double> params, double x, std::vector<double> y){
    double w = params[0];
    double gamma = params[1];
    double omega = params[2];
    double A = params[3];
    std::vector<double> f = {y[1], -w*w*y[0] - gamma*y[1] + A*std::sin(omega*x)};
    return f;
}

std::vector<double> DoublePendulum(std::vector<double> params, double t, std::vector<double> y) {
    double m1 = params[4], m2 = params[5], l1 = params[6], l2 = params[7], g = params[8], A = params[3], omega = params[2];
    
    double theta1 = y[0], theta2 = y[1], omega1 = y[2], omega2 = y[3];
    double delta = theta2 - theta1;
    
    double sin1 = sin(theta1), sin2 = sin(theta2);
    double sin_delta = sin(delta), cos_delta = cos(delta);
    double cos2_delta = cos(2 * delta);
    
    double numerator1 = m2 * l1 * omega1 * omega1 * sin_delta * cos_delta
                      + m2 * g * sin2 * cos_delta
                      + m2 * l2 * omega2 * omega2 * sin_delta
                      - (m1 + m2) * g * sin1;
    
    double denominator1 = (m1 + m2) * l1 - m2 * l1 * cos2_delta;
    
    double alpha1 = numerator1 / denominator1;
    
    double numerator2 = -m2 * l2 * omega2 * omega2 * sin_delta * cos_delta
                      + (m1 + m2) * (g * sin1 * cos_delta - l1 * omega1 * omega1 * sin_delta - g * sin2 - A * cos(omega*t));
    
    double denominator2 = (m1 + m2) * l2 - m2 * l2 * cos2_delta;
    
    double alpha2 = numerator2 / denominator2;
    
    return {omega1, omega2, alpha1, alpha2};
}

void RK4(PendulumFunc func, std::vector<double> params, double sim_time, double dt, 
         std::vector<std::vector<double>>& y, std::vector<double>& t) {
    const int steps = static_cast<int>(sim_time / dt);
    int dim = y[0].size();

    for (int i = 0; i < steps; i++) {
        std::vector<double> y_i = y.back();
        double t_i = t.back();

        //коэффициенты РК4
        std::vector<double> k1 = func(params, t_i, y_i);
        std::vector<double> k2 = func(params, t_i + dt/2, y_i + (dt/2) * k1);
        std::vector<double> k3 = func(params, t_i + dt/2, y_i + (dt/2) * k2);
        std::vector<double> k4 = func(params, t_i + dt, y_i + dt * k3);

        std::vector<double> y_new(dim);
        for (int j = 0; j < dim; j++) {
            y_new[j] = y_i[j] + (dt/6) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
        }
        
        y.push_back(y_new);
        t.push_back(t_i + dt);
    }
}

void Hoina(PendulumFunc func, std::vector<double> params, double sim_time, double dt, 
           std::vector<std::vector<double>>& y, std::vector<double>& t) {
    const int steps = static_cast<int>(sim_time / dt);
    int dim = y[0].size();

    for (int i = 0; i < steps; i++) {
        std::vector<double> y_i = y.back();
        double t_i = t.back();

        std::vector<double> k1 = func(params, t_i, y_i);
        
        // Предиктор
        std::vector<double> y_pred(dim);
        for (int j = 0; j < dim; j++) {
            y_pred[j] = y_i[j] + dt * k1[j];
        }
        
        // Корректор
        std::vector<double> k2 = func(params, t_i + dt, y_pred);
        
        std::vector<double> y_new(dim);
        for (int j = 0; j < dim; j++) {
            y_new[j] = y_i[j] + dt * 0.5 * (k1[j] + k2[j]);
        }
        
        y.push_back(y_new);
        t.push_back(t_i + dt);
    }
}

void Euler(PendulumFunc func, std::vector<double> params, double sim_time, double dt, 
           std::vector<std::vector<double>>& y, std::vector<double>& t) {
    const int steps = static_cast<int>(sim_time / dt);
    int dim = y[0].size();

    for (int i = 0; i < steps; i++) {
        std::vector<double> y_i = y.back();
        double t_i = t.back();

        std::vector<double> k = func(params, t_i, y_i);
        
        std::vector<double> y_new(dim);
        for (int j = 0; j < dim; j++) {
            y_new[j] = y_i[j] + dt * k[j];
        }
        
        y.push_back(y_new);
        t.push_back(t_i + dt);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    std::string c_path = argv[1];
    std::ifstream input_file(c_path);
    if (!input_file.is_open()) {
        std::cerr << "Cannot open config file: " << c_path << std::endl;
        return 1;
    }

    json j;
    input_file >> j;

    // Извлечение параметров
    double sim_time = j["simulation_time"];
    double dt = j["dt"];
    double v_0 = j["initial_velocity"];
    double x_0 = j["initial_position"];
    double w = j["w"];
    double gamma = j["gamma"];
    double omega = j["omega"];
    double A = j["A"];
    double m1 = j["m1"];
    double m2 = j["m2"];
    double l1 = j["l1"];
    double l2 = j["l2"];
    double g = j["g"];
    double theta1 = j["theta1"];
    double theta2 = j["theta2"];
    double omega1 = j["omega1"];
    double omega2 = j["omega2"];
    std::string output_path = j["output_file"];
    std::string method = j["method"];
    std::string equation = j["equation"];

    // Инициализация
    std::vector<double> params = {w, gamma, omega, A, m1, m2, l1, l2, g, theta1, theta2, omega1, omega2};
    std::vector<double> t = {0};
    std::vector<std::vector<double>> y;

    // Выбор уравнения и установка начальных условий
    PendulumFunc pendulum;
    if (equation == "MathPendulum") {
        pendulum = MathPendulum;
        y = {{x_0, v_0}};
    }
    else if (equation == "PhysPendulum") {
        pendulum = PhysPendulum;
        y = {{x_0, v_0}};
    }
    else if (equation == "DampedMathPendulum") {
        pendulum = DampedMathPendulum;
        y = {{x_0, v_0}};
    }
    else if (equation == "DampedPhysPendulum") {
        pendulum = DampedPhysPendulum;
        y = {{x_0, v_0}};
    }
    else if (equation == "ForcedDampedMathPendulum") {
        pendulum = ForcedDampedMathPendulum;
        y = {{x_0, v_0}};
    }
    else if (equation == "DoublePendulum") {
        pendulum = DoublePendulum;
        y = {{theta1, theta2, omega1, omega2}};
    }
    else {
        std::cerr << "Unknown equation type: " << equation << std::endl;
        return 1;
    }

    // Выбор метода
    if (method == "RK4") {
        RK4(pendulum, params, sim_time, dt, y, t);
    }
    else if (method == "Hoina") {
        Hoina(pendulum, params, sim_time, dt, y, t);
    }
    else if (method == "Euler") {
        Euler(pendulum, params, sim_time, dt, y, t);
    }
    else {
        std::cerr << "Unknown method: " << method << std::endl;
        return 1;
    }

    // Запись результатов
    std::ofstream fout(output_path);
    if (!fout.is_open()) {
        std::cerr << "Cannot open output file: " << output_path << std::endl;
        return 1;
    }

    if (equation == "DoublePendulum") {
        fout << "time,theta1,theta2,omega1,omega2" << std::endl;
        for (size_t i = 0; i < t.size(); i++) {
            fout << t[i] << "," << y[i][0] << "," << y[i][1] << "," 
                 << y[i][2] << "," << y[i][3] << std::endl;
        }
    } else {
        fout << "time,position,velocity,energy" << std::endl;
        for (size_t i = 0; i < t.size(); i++) {
            double energy = 0.5 * y[i][1] * y[i][1] + 0.5 * w * w * y[i][0] * y[i][0];
            fout << t[i] << "," << y[i][0] << "," << y[i][1] << "," << energy << std::endl;
        }
    }
    
    fout.close();
    return 0;
}