#pragma once

#include <sqlite3.h>

#include <stdexcept>
#include <string>

namespace flagfft {

class SqliteStmt;

class SqliteDb {
 public:
  explicit SqliteDb(const std::string &path);
  ~SqliteDb();
  SqliteDb(const SqliteDb &) = delete;
  SqliteDb &operator=(const SqliteDb &) = delete;

  void exec(const std::string &sql);

 private:
  friend class SqliteStmt;
  sqlite3 *db_ = nullptr;
};

class SqliteStmt {
 public:
  SqliteStmt(SqliteDb &db, const std::string &sql);
  ~SqliteStmt();
  SqliteStmt(const SqliteStmt &) = delete;
  SqliteStmt &operator=(const SqliteStmt &) = delete;

  void bind_int64(int index, int64_t value);
  void bind_text(int index, const std::string &value);
  void bind_double(int index, double value);
  void bind_null(int index);
  bool step();
  int64_t column_int64(int index);
  std::string column_text(int index);
  double column_double(int index);
  void reset();

 private:
  sqlite3_stmt *stmt_ = nullptr;
};

}  // namespace flagfft
