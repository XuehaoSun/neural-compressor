# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# flake8: noqa
# mypy: ignore-errors
"""v1.11

Revision ID: 9e89549a08c8
Revises: 
Create Date: 2022-03-22 13:34:52.916541

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "9e89549a08c8"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "tuning_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("minimal_accuracy", sa.Float(), nullable=True),
        sa.Column("baseline_accuracy", sa.String(), nullable=True),
        sa.Column("baseline_performance", sa.String(), nullable=True),
        sa.Column("last_tune_accuracy", sa.String(), nullable=True),
        sa.Column("last_tune_performance", sa.String(), nullable=True),
        sa.Column("best_tune_accuracy", sa.String(), nullable=True),
        sa.Column("best_tune_performance", sa.String(), nullable=True),
        sa.Column("history", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("modified_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tuning_history")),
    )
    with op.batch_alter_table("tuning_history", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_tuning_history_id"), ["id"], unique=True)

    op.create_table(
        "example",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("framework", sa.Integer(), nullable=False),
        sa.Column("domain", sa.Integer(), nullable=False),
        sa.Column("dataset_type", sa.String(length=50), nullable=False),
        sa.Column("model_url", sa.String(length=250), nullable=False),
        sa.Column("config_url", sa.String(length=250), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["domain"], ["domain.id"], name=op.f("fk_example_domain_domain")),
        sa.ForeignKeyConstraint(
            ["framework"], ["framework.id"], name=op.f("fk_example_framework_framework")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_example")),
    )
    with op.batch_alter_table("example", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_example_id"), ["id"], unique=False)

    with op.batch_alter_table("benchmark_result", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            batch_op.f("uq_benchmark_result_benchmark_id"), ["benchmark_id"]
        )

    with op.batch_alter_table("domain", schema=None) as batch_op:
        batch_op.create_unique_constraint(batch_op.f("uq_domain_name"), ["name"])

    with op.batch_alter_table("domain_flavour", schema=None) as batch_op:
        batch_op.create_unique_constraint(batch_op.f("uq_domain_flavour_name"), ["name"])

    with op.batch_alter_table("framework", schema=None) as batch_op:
        batch_op.create_unique_constraint(batch_op.f("uq_framework_name"), ["name"])

    with op.batch_alter_table("optimization_type", schema=None) as batch_op:
        batch_op.create_unique_constraint(batch_op.f("uq_optimization_type_name"), ["name"])

    with op.batch_alter_table("precision", schema=None) as batch_op:
        batch_op.create_unique_constraint(batch_op.f("uq_precision_name"), ["name"])

    with op.batch_alter_table("tuning_details", schema=None) as batch_op:
        batch_op.add_column(sa.Column("tuning_history_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            batch_op.f("fk_tuning_details_tuning_history_id_tuning_history"),
            "tuning_history",
            ["tuning_history_id"],
            ["id"],
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("tuning_details", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("fk_tuning_details_tuning_history_id_tuning_history"), type_="foreignkey"
        )
        batch_op.drop_column("tuning_history_id")

    with op.batch_alter_table("precision", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("uq_precision_name"), type_="unique")

    with op.batch_alter_table("optimization_type", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("uq_optimization_type_name"), type_="unique")

    with op.batch_alter_table("framework", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("uq_framework_name"), type_="unique")

    with op.batch_alter_table("domain_flavour", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("uq_domain_flavour_name"), type_="unique")

    with op.batch_alter_table("domain", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("uq_domain_name"), type_="unique")

    with op.batch_alter_table("benchmark_result", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("uq_benchmark_result_benchmark_id"), type_="unique")

    with op.batch_alter_table("example", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_example_id"))

    op.drop_table("example")
    with op.batch_alter_table("tuning_history", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_tuning_history_id"))

    op.drop_table("tuning_history")
    # ### end Alembic commands ###