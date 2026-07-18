// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
