package org.lenskit.data.events;

import org.apache.commons.lang3.ClassUtils;
import org.grouplens.grapht.util.ClassLoaders;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Enumeration;
import java.util.Properties;

/**
 * Look up event types.  This class looks for `META-INF/lenskit/event-builders.properties` files in the classpath and
 * looks up event type names in them.
 */
public class EventTypeResolver {
    private static final Logger logger = LoggerFactory.getLogger(EventTypeResolver.class);
    private final ClassLoader classLoader;
    private final Properties typeDefs;

    EventTypeResolver(ClassLoader loader) {
        classLoader = loader;
        typeDefs = new Properties();
        try {
            Enumeration<URL> files = classLoader.getResources("META-INF/lenskit/event-builders.properties");
            while (files.hasMoreElements()) {
                URL url = files.nextElement();
                try (InputStream str = url.openStream()) {
                    logger.debug("loading {}", url);
                    typeDefs.load(str);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("cannot scan for event type files", e);
        }
    }

    /**
     * Create a new event type resolver for a class loader.
     * @param loader The class loader.
     * @return The event type resolver.
     */
    public static EventTypeResolver create(ClassLoader loader) {
        return new EventTypeResolver(loader);
    }

    /**
     * Create a new event type resolver for the current class loader.
     * @return The event type resolver.
     * @see ClassLoaders#inferDefault()
     */
    public static EventTypeResolver create() {
        return new EventTypeResolver(ClassLoaders.inferDefault(EventTypeResolver.class));
    }

    /**
     * Get an event builder for the specified type name.  It first looks up the type name using the properties
     * files loaded from the classpath, then tries to instantiate it as a class.
     * @param name The type name.
     * @return The event builder.
     */
    @Nullable
    @SuppressWarnings("unchecked")
    public EventBuilder<?> getEventBuilder(String name) {
        String className = typeDefs.getProperty(name);
        if (className == null) {
            className = name;
        }

        try {
            Class<? extends EventBuilder<?>> cls =
                    (Class) ClassUtils.getClass(classLoader, className).asSubclass(EventBuilder.class);
            return cls.newInstance();
        } catch (ClassNotFoundException e) {
            logger.debug("cannot locate class {}", className);
            return null;
        } catch (InstantiationException | IllegalAccessException e) {
            throw new RuntimeException("cannot instantiate " + className, e);
        }
    }

    @Nullable
    @SuppressWarnings("unchecked")
    public <E extends Event> EventBuilder<E> getEventBuilder(Class<E> eventType) {
        BuiltBy bb = eventType.getAnnotation(BuiltBy.class);
        if (bb == null) {
            return null;
        }
        Class<? extends EventBuilder<?>> cls = bb.value();
        try {
            return EventBuilder.class.cast(cls.newInstance());
        } catch (InstantiationException | IllegalAccessException e) {
            throw new RuntimeException("cannot instantiate " + cls, e);
        }
    }
}
