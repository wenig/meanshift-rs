use actix::{Actor, Context, System};
use actix::io::SinkWrite;
use crate::meanshift_actors::interface::MySink;


pub struct SinkActor<T> where T: Unpin + Clone {
    pub sink: SinkWrite<T, MySink<T>>,
    pub result: Option<T>
}

impl<T> SinkActor<T> where T: Unpin + Clone {
    pub fn new(sink: SinkWrite<T, MySink<T>>) -> Self {
        Self {
            sink,
            result: None
        }
    }
}

impl<T> Actor for SinkActor<T> where T: Unpin + 'static + Clone {
    type Context = Context<SinkActor<T>>;
}

impl<T> actix::io::WriteHandler<()> for SinkActor<T> where T: Unpin + 'static + Clone {
    fn finished(&mut self, _ctxt: &mut Self::Context) {
        System::current().stop();
    }
}
