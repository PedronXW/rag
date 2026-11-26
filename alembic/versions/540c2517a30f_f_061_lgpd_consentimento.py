"""F-061 lgpd: consentimento do usuário (consent_granted/consent_at)

Revision ID: 540c2517a30f
Revises: c19610cbe5e6
Create Date: 2026-11-26 13:20:00

Adiciona colunas de consentimento LGPD em `users` sem tocar nas tabelas de
checkpoint do LangGraph (geridas pelo runtime).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '540c2517a30f'
down_revision: Union[str, Sequence[str], None] = 'c19610cbe5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('users', sa.Column('consent_granted', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('users', sa.Column('consent_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('users', 'consent_at')
    op.drop_column('users', 'consent_granted')