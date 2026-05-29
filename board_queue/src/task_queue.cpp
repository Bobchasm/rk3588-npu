#include "rkq/task_queue.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <pwd.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

namespace rkq {
namespace {

volatile sig_atomic_t g_signal_flag = 0;
volatile sig_atomic_t g_child_pid = -1;

void signal_handler(int signo) {
    g_signal_flag = signo;
    if (g_child_pid > 0) {
        kill(g_child_pid, signo);
    }
}

std::string current_user_name() {
    const char* user = std::getenv("USER");
    if (user && user[0] != '\0') {
        return user;
    }
    struct passwd* pwd = getpwuid(getuid());
    if (pwd && pwd->pw_name) {
        return pwd->pw_name;
    }
    return "unknown";
}

std::string current_host_name() {
    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname) - 1) == 0) {
        return hostname;
    }
    return "unknown";
}

std::string current_working_dir() {
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    return ec ? std::string(".") : cwd.string();
}

std::string read_whole_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        return {};
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

void write_whole_file(const std::string& path, const std::string& content) {
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("failed to open file for write: " + path);
    }
    ofs << content;
}

std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

}  // namespace

TaskQueueManager::TaskQueueManager(const std::string& root_dir)
    : root_dir_(root_dir),
      queue_dir_(root_dir_ + "/queue"),
      finished_dir_(root_dir_ + "/finished"),
      current_file_(root_dir_ + "/current.task"),
      history_file_(root_dir_ + "/history.log"),
      state_lock_file_(root_dir_ + "/state.lock"),
      state_lock_fd_(-1) {
    ensure_layout();
    state_lock_fd_ = ::open(state_lock_file_.c_str(), O_CREAT | O_RDWR, 0666);
    if (state_lock_fd_ < 0) {
        throw std::runtime_error("failed to open state lock file: " + state_lock_file_);
    }
}

TaskQueueManager::~TaskQueueManager() {
    if (state_lock_fd_ >= 0) {
        ::close(state_lock_fd_);
        state_lock_fd_ = -1;
    }
}

void TaskQueueManager::ensure_layout() const {
    std::error_code ec;
    fs::create_directories(queue_dir_, ec);
    fs::create_directories(finished_dir_, ec);
    std::ofstream(history_file_, std::ios::app).close();
}

void TaskQueueManager::lock_state() const {
    if (flock(state_lock_fd_, LOCK_EX) != 0) {
        throw std::runtime_error("failed to lock state");
    }
}

void TaskQueueManager::unlock_state() const {
    flock(state_lock_fd_, LOCK_UN);
}

std::string TaskQueueManager::status_to_string(TaskStatus status) {
    switch (status) {
    case TaskStatus::kQueued: return "queued";
    case TaskStatus::kRunning: return "running";
    case TaskStatus::kFinished: return "finished";
    case TaskStatus::kFailed: return "failed";
    case TaskStatus::kCanceled: return "canceled";
    default: return "unknown";
    }
}

TaskStatus TaskQueueManager::string_to_status(const std::string& text) {
    if (text == "queued") return TaskStatus::kQueued;
    if (text == "running") return TaskStatus::kRunning;
    if (text == "finished") return TaskStatus::kFinished;
    if (text == "failed") return TaskStatus::kFailed;
    if (text == "canceled") return TaskStatus::kCanceled;
    return TaskStatus::kQueued;
}

std::string TaskQueueManager::now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&tt, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

long TaskQueueManager::now_epoch_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string TaskQueueManager::shell_escape(const std::string& value) {
    // 仅用于日志展示，不参与实际 exec，因此保留简单直观的 shell 逃逸。
    std::string escaped = "'";
    for (char ch : value) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

std::string TaskQueueManager::join_command(const std::vector<std::string>& argv) {
    std::ostringstream oss;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i != 0) {
            oss << ' ';
        }
        oss << shell_escape(argv[i]);
    }
    return oss.str();
}

std::string TaskQueueManager::join_tab_escaped(const std::vector<std::string>& parts) {
    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            oss << '\t';
        }
        for (char ch : parts[i]) {
            if (ch == '\\') {
                oss << "\\\\";
            } else if (ch == '\t') {
                oss << "\\t";
            } else if (ch == '\n') {
                oss << "\\n";
            } else {
                oss << ch;
            }
        }
    }
    return oss.str();
}

std::vector<std::string> TaskQueueManager::split_tab_escaped(const std::string& value) {
    std::vector<std::string> result;
    std::string current;
    bool escaped = false;
    for (char ch : value) {
        if (escaped) {
            if (ch == 't') current.push_back('\t');
            else if (ch == 'n') current.push_back('\n');
            else current.push_back(ch);
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '\t') {
            result.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    result.push_back(current);
    return result;
}

TaskRecord TaskQueueManager::build_task_record(const std::vector<std::string>& argv) const {
    TaskRecord task;
    task.submit_epoch_ms = now_epoch_ms();
    task.submit_time = now_iso8601();
    task.user_name = current_user_name();
    task.host_name = current_host_name();
    task.working_dir = current_working_dir();
    task.command_argv = argv;
    task.command_display = join_command(argv);
    task.submit_pid = static_cast<int>(getpid());
    task.status = TaskStatus::kQueued;

    std::ostringstream task_id;
    task_id << task.submit_epoch_ms << "_" << task.user_name << "_" << task.submit_pid;
    task.task_id = task_id.str();
    return task;
}

std::string TaskQueueManager::make_task_file_path(const std::string& task_id) const {
    return queue_dir_ + "/" + task_id + ".task";
}

std::string TaskQueueManager::make_finished_file_path(const std::string& task_id) const {
    return finished_dir_ + "/" + task_id + ".task";
}

void TaskQueueManager::write_task_file(const std::string& path, const TaskRecord& task) const {
    std::ostringstream oss;
    oss << "task_id=" << task.task_id << "\n";
    oss << "user_name=" << task.user_name << "\n";
    oss << "host_name=" << task.host_name << "\n";
    oss << "submit_time=" << task.submit_time << "\n";
    oss << "start_time=" << task.start_time << "\n";
    oss << "finish_time=" << task.finish_time << "\n";
    oss << "working_dir=" << task.working_dir << "\n";
    oss << "command_display=" << task.command_display << "\n";
    oss << "command_argv=" << join_tab_escaped(task.command_argv) << "\n";
    oss << "submit_epoch_ms=" << task.submit_epoch_ms << "\n";
    oss << "start_epoch_ms=" << task.start_epoch_ms << "\n";
    oss << "finish_epoch_ms=" << task.finish_epoch_ms << "\n";
    oss << "submit_pid=" << task.submit_pid << "\n";
    oss << "exit_code=" << task.exit_code << "\n";
    oss << "cancel_requested=" << task.cancel_requested << "\n";
    oss << "status=" << status_to_string(task.status) << "\n";
    write_whole_file(path, oss.str());
}

bool TaskQueueManager::read_task_file(const std::string& path, TaskRecord& task) const {
    std::ifstream ifs(path);
    if (!ifs) {
        return false;
    }
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(ifs, line)) {
        const auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        kv[line.substr(0, pos)] = line.substr(pos + 1);
    }
    task.task_id = kv["task_id"];
    task.user_name = kv["user_name"];
    task.host_name = kv["host_name"];
    task.submit_time = kv["submit_time"];
    task.start_time = kv["start_time"];
    task.finish_time = kv["finish_time"];
    task.working_dir = kv["working_dir"];
    task.command_display = kv["command_display"];
    task.command_argv = split_tab_escaped(kv["command_argv"]);
    task.submit_epoch_ms = std::stol(kv["submit_epoch_ms"].empty() ? "0" : kv["submit_epoch_ms"]);
    task.start_epoch_ms = std::stol(kv["start_epoch_ms"].empty() ? "0" : kv["start_epoch_ms"]);
    task.finish_epoch_ms = std::stol(kv["finish_epoch_ms"].empty() ? "0" : kv["finish_epoch_ms"]);
    task.submit_pid = std::stoi(kv["submit_pid"].empty() ? "0" : kv["submit_pid"]);
    task.exit_code = std::stoi(kv["exit_code"].empty() ? "0" : kv["exit_code"]);
    task.cancel_requested = std::stoi(kv["cancel_requested"].empty() ? "0" : kv["cancel_requested"]);
    task.status = string_to_status(kv["status"]);
    return true;
}

bool TaskQueueManager::load_current_task(TaskRecord& task) const {
    if (!fs::exists(current_file_)) {
        return false;
    }
    return read_task_file(current_file_, task);
}

std::vector<TaskRecord> TaskQueueManager::load_queued_tasks() const {
    std::vector<TaskRecord> tasks;
    if (!fs::exists(queue_dir_)) {
        return tasks;
    }
    for (const auto& entry : fs::directory_iterator(queue_dir_)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        TaskRecord task;
        if (read_task_file(entry.path().string(), task)) {
            tasks.push_back(task);
        }
    }
    std::sort(tasks.begin(), tasks.end(), [](const TaskRecord& a, const TaskRecord& b) {
        return a.task_id < b.task_id;
    });
    return tasks;
}

void TaskQueueManager::append_history_line(const TaskRecord& task, const std::string& event) const {
    std::ofstream ofs(history_file_, std::ios::app);
    ofs << now_iso8601()
        << " event=" << event
        << " task_id=" << task.task_id
        << " user=" << task.user_name
        << " status=" << status_to_string(task.status)
        << " exit_code=" << task.exit_code
        << " cmd=" << task.command_display
        << "\n";
}

void TaskQueueManager::update_current_snapshot(const TaskRecord& task) const {
    write_task_file(current_file_, task);
}

void TaskQueueManager::clear_current_snapshot() const {
    std::error_code ec;
    fs::remove(current_file_, ec);
}

void TaskQueueManager::move_task_to_finished(const TaskRecord& task) const {
    write_task_file(make_finished_file_path(task.task_id), task);
}

bool TaskQueueManager::try_promote_to_running(const TaskRecord& self) {
    lock_state();
    const auto queued = load_queued_tasks();
    if (queued.empty()) {
        unlock_state();
        return false;
    }
    if (queued.front().task_id != self.task_id) {
        unlock_state();
        return false;
    }
    if (fs::exists(current_file_)) {
        unlock_state();
        return false;
    }

    TaskRecord running = self;
    running.status = TaskStatus::kRunning;
    running.start_epoch_ms = now_epoch_ms();
    running.start_time = now_iso8601();
    update_current_snapshot(running);
    write_task_file(make_task_file_path(self.task_id), running);
    append_history_line(running, "start");
    unlock_state();
    return true;
}

int TaskQueueManager::wait_until_turn(const TaskRecord& self) const {
    while (true) {
        if (g_signal_flag != 0) {
            return 128 + g_signal_flag;
        }

        TaskRecord latest;
        if (read_task_file(make_task_file_path(self.task_id), latest) && latest.cancel_requested != 0) {
            std::cerr << "[rkq] task was canceled before execution: " << self.task_id << "\n";
            return 130;
        }

        lock_state();
        const auto queued = load_queued_tasks();
        const bool has_current = fs::exists(current_file_);
        int position = -1;
        for (size_t i = 0; i < queued.size(); ++i) {
            if (queued[i].task_id == self.task_id) {
                position = static_cast<int>(i) + 1;
                break;
            }
        }
        unlock_state();

        if (position < 0) {
            std::cerr << "[rkq] queued task disappeared unexpectedly: " << self.task_id << "\n";
            return 1;
        }

        // 只有当前任务排在队首且当前没有其它运行任务时，才允许外层再次尝试抢占执行权。
        if (position == 1 && !has_current) {
            std::cerr << "\r[rkq] waiting in queue, position=1 task_id="
                      << self.task_id << " -> ready to run           \n";
            return 0;
        }

        std::cerr << "\r[rkq] waiting in queue, position=" << position
                  << " task_id=" << self.task_id << "   " << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int TaskQueueManager::execute_task(TaskRecord& self) {
    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);

    pid_t child = fork();
    if (child < 0) {
        throw std::runtime_error("fork failed");
    }

    if (child == 0) {
        if (chdir(self.working_dir.c_str()) != 0) {
            std::perror("chdir");
            _exit(127);
        }

        std::vector<char*> argv_ptrs;
        argv_ptrs.reserve(self.command_argv.size() + 1);
        for (std::string& arg : self.command_argv) {
            argv_ptrs.push_back(arg.data());
        }
        argv_ptrs.push_back(nullptr);
        execvp(argv_ptrs[0], argv_ptrs.data());
        std::perror("execvp");
        _exit(127);
    }

    g_child_pid = child;
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        throw std::runtime_error("waitpid failed");
    }
    g_child_pid = -1;

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 1;
}

int TaskQueueManager::run_command(const std::vector<std::string>& argv) {
    if (argv.empty()) {
        std::cerr << "rkq run requires a command after '--'\n";
        return 2;
    }

    TaskRecord self = build_task_record(argv);
    const std::string task_path = make_task_file_path(self.task_id);

    lock_state();
    write_task_file(task_path, self);
    append_history_line(self, "enqueue");
    unlock_state();

    std::cerr << "[rkq] enqueued task_id=" << self.task_id
              << " user=" << self.user_name
              << " cwd=" << self.working_dir << "\n";

    while (!try_promote_to_running(self)) {
        const int wait_rc = wait_until_turn(self);
        if (wait_rc != 0) {
            lock_state();
            TaskRecord latest;
            if (read_task_file(task_path, latest)) {
                latest.status = (latest.cancel_requested != 0) ? TaskStatus::kCanceled : TaskStatus::kFailed;
                latest.finish_epoch_ms = now_epoch_ms();
                latest.finish_time = now_iso8601();
                move_task_to_finished(latest);
                append_history_line(latest, latest.cancel_requested != 0 ? "cancel" : "abort_wait");
            }
            std::error_code ec;
            fs::remove(task_path, ec);
            unlock_state();
            return wait_rc;
        }
    }

    std::cerr << "\n[rkq] task is now running: " << self.task_id << "\n";
    read_task_file(task_path, self);
    const int rc = execute_task(self);

    lock_state();
    TaskRecord final_task;
    if (!read_task_file(task_path, final_task)) {
        final_task = self;
    }
    final_task.exit_code = rc;
    final_task.finish_epoch_ms = now_epoch_ms();
    final_task.finish_time = now_iso8601();
    final_task.status = (rc == 0) ? TaskStatus::kFinished : TaskStatus::kFailed;
    move_task_to_finished(final_task);
    clear_current_snapshot();
    append_history_line(final_task, (rc == 0) ? "finish" : "fail");
    std::error_code ec;
    fs::remove(task_path, ec);
    unlock_state();

    std::cerr << "[rkq] task finished task_id=" << final_task.task_id
              << " exit_code=" << rc
              << " elapsed_ms=" << (final_task.finish_epoch_ms - final_task.start_epoch_ms)
              << "\n";
    return rc;
}

int TaskQueueManager::print_status() const {
    lock_state();
    TaskRecord current;
    const bool has_current = load_current_task(current);
    const auto queued = load_queued_tasks();
    unlock_state();

    std::cout << "Current:\n";
    if (!has_current) {
        std::cout << "  (none)\n";
    } else {
        std::cout << "  task_id: " << current.task_id << "\n"
                  << "  user:    " << current.user_name << "\n"
                  << "  status:  " << status_to_string(current.status) << "\n"
                  << "  start:   " << current.start_time << "\n"
                  << "  cwd:     " << current.working_dir << "\n"
                  << "  cmd:     " << current.command_display << "\n";
    }

    std::cout << "\nQueue:\n";
    if (queued.empty()) {
        std::cout << "  (empty)\n";
    } else {
        for (size_t i = 0; i < queued.size(); ++i) {
            std::cout << "  " << (i + 1) << ". "
                      << queued[i].task_id
                      << " user=" << queued[i].user_name
                      << " status=" << status_to_string(queued[i].status)
                      << " submit=" << queued[i].submit_time
                      << " cmd=" << queued[i].command_display
                      << "\n";
        }
    }
    return 0;
}

int TaskQueueManager::print_current() const {
    lock_state();
    TaskRecord current;
    const bool has_current = load_current_task(current);
    unlock_state();

    if (!has_current) {
        std::cout << "No running task.\n";
        return 0;
    }

    std::cout << "task_id=" << current.task_id << "\n"
              << "user=" << current.user_name << "\n"
              << "status=" << status_to_string(current.status) << "\n"
              << "start_time=" << current.start_time << "\n"
              << "working_dir=" << current.working_dir << "\n"
              << "command=" << current.command_display << "\n";
    return 0;
}

int TaskQueueManager::cancel_task(const std::string& task_id) {
    if (task_id.empty()) {
        std::cerr << "rkq cancel requires a task_id\n";
        return 2;
    }

    const std::string path = make_task_file_path(task_id);
    lock_state();
    TaskRecord task;
    if (!read_task_file(path, task)) {
        unlock_state();
        std::cerr << "task not found in queue: " << task_id << "\n";
        return 1;
    }

    if (task.status == TaskStatus::kRunning) {
        unlock_state();
        std::cerr << "task is already running, first version does not force-kill running tasks\n";
        return 1;
    }

    task.cancel_requested = 1;
    task.status = TaskStatus::kCanceled;
    task.finish_epoch_ms = now_epoch_ms();
    task.finish_time = now_iso8601();
    move_task_to_finished(task);
    append_history_line(task, "cancel");
    std::error_code ec;
    fs::remove(path, ec);
    unlock_state();

    std::cout << "Canceled task: " << task_id << "\n";
    return 0;
}

}  // namespace rkq
