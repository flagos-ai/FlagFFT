#include "sqlite_wrapper.hpp"

namespace flagfft {

SqliteDb::SqliteDb(const std::string &path) {
    int rc = sqlite3_open(path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string msg = sqlite3_errmsg(db_);
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("failed to open sqlite database: " + msg);
    }
}

SqliteDb::~SqliteDb() {
    if (db_ != nullptr) {
        sqlite3_close(db_);
    }
}

void SqliteDb::exec(const std::string &sql) {
    char *err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::string msg = err_msg != nullptr ? err_msg : "unknown error";
        sqlite3_free(err_msg);
        throw std::runtime_error("sqlite exec failed: " + msg);
    }
}

SqliteStmt::SqliteStmt(SqliteDb &db, const std::string &sql) {
    int rc = sqlite3_prepare_v2(db.db_, sql.c_str(), -1, &stmt_, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("sqlite prepare failed: " +
                                 std::string(sqlite3_errmsg(db.db_)));
    }
}

SqliteStmt::~SqliteStmt() {
    if (stmt_ != nullptr) {
        sqlite3_finalize(stmt_);
    }
}

void SqliteStmt::bind_int64(int index, int64_t value) {
    sqlite3_bind_int64(stmt_, index, value);
}

void SqliteStmt::bind_text(int index, const std::string &value) {
    sqlite3_bind_text(stmt_, index, value.c_str(), -1, SQLITE_TRANSIENT);
}

void SqliteStmt::bind_double(int index, double value) {
    sqlite3_bind_double(stmt_, index, value);
}

void SqliteStmt::bind_null(int index) {
    sqlite3_bind_null(stmt_, index);
}

bool SqliteStmt::step() {
    int rc = sqlite3_step(stmt_);
    if (rc == SQLITE_ROW) {
        return true;
    }
    if (rc == SQLITE_DONE) {
        return false;
    }
    throw std::runtime_error("sqlite step failed: " +
                             std::string(sqlite3_errmsg(sqlite3_db_handle(stmt_))));
}

int64_t SqliteStmt::column_int64(int index) {
    return sqlite3_column_int64(stmt_, index);
}

std::string SqliteStmt::column_text(int index) {
    const unsigned char *text = sqlite3_column_text(stmt_, index);
    if (text == nullptr) {
        return "";
    }
    return std::string(reinterpret_cast<const char *>(text));
}

double SqliteStmt::column_double(int index) {
    return sqlite3_column_double(stmt_, index);
}

void SqliteStmt::reset() {
    sqlite3_reset(stmt_);
}

}  // namespace flagfft
