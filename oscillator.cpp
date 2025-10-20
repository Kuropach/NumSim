#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using PendulumFunc = std::vector<double>(*)(std::vector<double>, std::vector<double>);

std::vector<double> MathPendulum(std::vector<double> params, std::vector<double> x){
    double w = params[0];
    std::vector<double> y = {x[1], -w*w*x[0]};
    return y;
}
std::vector<double> PhysPendulum(std::vector<double> params, std::vector<double> x){
    double w = params[0];
    std::vector<double> y = {x[1], -w*w*std::sin(x[0])};
    return y;
}
std::vector<double> DampedMathPendulum(std::vector<double> params, std::vector<double> x){
    double w = params[0];
    double gamma = params[1];
    std::vector<double> y = {x[1], -w*w*x[0] - gamma*x[1]};
    return y;
}
std::vector<double> DampedPhysPendulum(std::vector<double> params, std::vector<double> x){
    double w = params[0];
    double gamma = params[1];
    std::vector<double> y = {x[1], -w*w*std::sin(x[0]) - gamma*x[1]};
    return y;
}

void RK4(PendulumFunc func, std::vector<double> params, double sim_time, double dt, std::vector<double> &x, std::vector<double> &v, std::vector<double> &t){
    const int steps = static_cast<int>(sim_time / dt);

    for (int i = 0; i < steps; i++)  {
        double x_i = x.back();
        double v_i = v.back();

        std::vector<double> y1 = func(params, {x_i, v_i});
        std::vector<double> y2 = func(params, {x_i + dt/2*y1[0], v_i + dt/2*y1[1]});
        std::vector<double> y3 = func(params, {x_i + dt/2*y2[0], v_i + dt/2*y2[1]});
        std::vector<double> y4 = func(params, {x_i + dt*y3[0], v_i + dt*y3[1]});

        x.push_back(x[i] + dt/6*(y1[0] + 2*y2[0] + 2*y3[0] + y4[0]));
        v.push_back(v[i] + dt/6*(y1[1] + 2*y2[1] + 2*y3[1] + y4[1]));
        t.push_back(t[i]+dt);
    }
}

void Hoina(PendulumFunc func, std::vector<double> params, double sim_time, double dt, std::vector<double> &x, std::vector<double> &v, std::vector<double> &t){
    const int steps = static_cast<int>(sim_time / dt);

    for (int i = 0; i < steps; i++)  {
        double x_i = x.back();
        double v_i = v.back();

        std::vector<double> y1 = func(params, {x_i, v_i}); //считаем пердикторы
        std::vector<double> y2 = func(params, {x_i + y1[0], v_i + y1[1]}); //считаем корректор

        double x_next = x_i + dt * 0.5 * y2[0];
        double v_next = v_i + dt * 0.5 * y2[1];
        double t_next = t.back() + dt;
        
        x.push_back(x_next);
        v.push_back(v_next);
        t.push_back(t_next);
    }
}

void Euler(PendulumFunc func, std::vector<double> params, double sim_time, double dt, std::vector<double> &x, std::vector<double> &v, std::vector<double> &t){
    const int steps = static_cast<int>(sim_time / dt);

    for (int i = 0; i < steps; i++)  {
        double x_i = x.back();
        double v_i = v.back();

        std::vector<double> y = func(params, {x_i, v_i});

        double x_next = x_i + dt * y[0];
        double v_next = v_i + dt * y[1];
        double t_next = t.back() + dt;

        x.push_back(x_next);
        v.push_back(v_next);
        t.push_back(t_next);
    }
}

int main(int argc, char* argv[]) {
    std::string c_path = argv[1];
	std::ifstream input_file(c_path);

    // Чтение JSON файла конфигурации
	json j;
	input_file >> j;

    // Извлечение параметров из JSON объекта
	double sim_time = j["simulation_time"];
	double dt = j["dt"];
	double v_0 = j["initial_velocity"];
	double x_0 = j["initial_position"];
	double w = j["w"];
	double gamma = j["gamma"];
	std::string output_path = j["output_file"];
    std::string method = j["method"];
    std::string equation = j["equation"];

    // Инициализация векторов для хранения результатов
    std::vector<double> params = {w, gamma};
    std::vector<double> x = {x_0};
    std::vector<double> v = {v_0};
    std::vector<double> t = {0};

    PendulumFunc pendulum;
    if (equation == "MathPendulum")
        pendulum = MathPendulum;
    else if (equation == "PhysPendulum")
        pendulum = PhysPendulum;
    else if (equation == "DampedMathPendulum")
        pendulum = DampedMathPendulum;
    else if (equation == "DampedPhysPendulum")
        pendulum = DampedPhysPendulum;
    else {
        std::cerr << "Unknown equation type!" << std::endl;
        return 1;
    }

    if (method == "RK4")
        RK4(pendulum, params, sim_time, dt, x, v, t);
    if (method == "Hoina")
        Hoina(pendulum, params, sim_time, dt, x, v, t);
    if (method == "Euler")
        Euler(pendulum, params, sim_time, dt, x, v, t);

    // Открытие файла для записи результатов
	std::ofstream fout;
	fout.open(output_path);
    //Запись
    fout << "time,position,velocity,energy" << std::endl;
    for (int i=0; i<= (sim_time / dt);  i++){
        fout << t[i] << "," << x[i] << "," << v[i] << "," << (v[i]*v[i]/2 + w*w*x[i]*x[i]/2) << std::endl;
    }
	fout.close();
    return 0;
}