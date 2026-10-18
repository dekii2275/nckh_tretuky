PYTHON ?= python3

.PHONY: help check test test-bilstm test-dryad test-audit audit-pipeline verify-pipeline

help:
	@printf 'Lệnh chung (chạy từ repo root):\n'
	@printf '  make check         Kiểm tra cú pháp Python, link README/tài liệu\n'
	@printf '  make test-bilstm   Chạy kiểm thử MMASD\n'
	@printf '  make test-dryad    Chạy kiểm thử DRYAD\n'
	@printf '  make test          Chạy cả hai nhóm kiểm thử\n'
	@printf '  make test-audit    Kiểm thử pipeline audit; không cần dataset/checkpoint\n'
	@printf '  make audit-pipeline  Tạo snapshot audit v0.2; cần dữ liệu cục bộ\n'
	@printf '  make verify-pipeline Kiểm tra snapshot audit v0.2\n'
	@printf 'Dùng PYTHON=/path/to/python để chọn môi trường.\n'

check:
	$(PYTHON) tools/check_repo.py

test-bilstm:
	$(PYTHON) -m pytest -q BiLSTM/tests

test-dryad:
	$(PYTHON) -m pytest -q tests

test: test-bilstm test-dryad

test-audit:
	$(PYTHON) -m pytest -q BiLSTM/tests/test_pipeline_lock.py

audit-pipeline:
	$(PYTHON) BiLSTM/scripts/audit_pipeline_lock.py

verify-pipeline:
	$(PYTHON) BiLSTM/scripts/audit_pipeline_lock.py --verify
