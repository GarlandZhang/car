from . import db
class FaceEncoding(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(50), unique=True, nullable=False)
  encoding = db.Column(db.String, unique=False, nullable=False)

  def __repr__(self):
    return '<FaceEncoding %s>' % self.name
