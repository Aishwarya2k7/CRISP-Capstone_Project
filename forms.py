from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
class EntryForm(FlaskForm):
    videoid = StringField(label='videoid')
    numlines = IntegerField(label='numlines')
    submit = SubmitField(label='submit')
